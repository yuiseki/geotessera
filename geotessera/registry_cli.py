#!/usr/bin/env python3
"""
Command-line interface for managing GeoTessera registry files.

This module provides tools for generating and maintaining Pooch registry files
used by the GeoTessera package. It supports parallel processing, incremental
updates, and generation of a master registry index.
"""

import os
import hashlib
import argparse
import subprocess
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Any, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from .registry import (
    block_from_world,
    block_to_embeddings_registry_filename,
    block_to_landmasks_registry_filename,
    parse_grid_name,
)

# Module-level logger
logger = logging.getLogger(__name__)

# Create console with automatic terminal detection
console = Console()


def emoji(text):
    """Return emoji text for smart terminals, empty string for dumb/piped output.

    Uses Rich Console's built-in terminal detection plus additional checks
    for dumb terminals and Windows legacy console encoding issues.
    """
    import os
    import sys

    # Check for dumb terminal
    if os.environ.get("TERM", "").lower() == "dumb":
        return ""

    # Check for Windows legacy console with cp1252 encoding
    if sys.platform == "win32":
        try:
            encoding = sys.stdout.encoding or ""
            if encoding.lower() in ("cp1252", "ascii", ""):
                return ""
        except Exception:
            return ""

    return text if console.is_terminal else ""


@dataclass
class TileInfo:
    """Complete information about a single tessera tile."""

    year: int
    lat: float
    lon: float
    grid_name: str
    embedding_path: str
    scales_path: str
    directory_path: str
    embedding_hash: Optional[str]
    scales_hash: Optional[str]
    embedding_mtime: float
    scales_mtime: float
    embedding_size: int
    scales_size: int


def process_grid_directory(args):
    """Process a single grid directory and return TileInfo.

    Args:
        args: Tuple of (year, year_path, grid_item, base_dir)

    Returns:
        TileInfo object or None if directory should be skipped
    """
    year, year_path, grid_item, base_dir = args
    grid_path = os.path.join(year_path, grid_item)

    try:
        # Check if directory has any files at all
        dir_contents = os.listdir(grid_path)
        if not dir_contents:
            return None  # Empty directory - skip silently

        # Check if this looks like a tile directory (has SHA256 or .npy files)
        has_sha256 = "SHA256" in dir_contents
        has_npy_files = any(f.endswith(".npy") for f in dir_contents)

        if not has_sha256 and not has_npy_files:
            return None  # Directory has files but doesn't look like a tile directory - skip

        # Parse coordinates from grid name
        lon, lat = parse_grid_name(grid_item)
        if lon is None or lat is None:
            raise ValueError(f"Could not parse coordinates from grid name: {grid_item}")

        # Read SHA256 file
        sha256_file = os.path.join(grid_path, "SHA256")
        if not os.path.exists(sha256_file):
            # If there are .npy files but no SHA256, log warning and skip
            if has_npy_files:
                logger.warning(f"Skipping directory without SHA256 file: {sha256_file}")
            return None  # Skip this directory

        # Parse hashes from SHA256 file
        embedding_hash = None
        scales_hash = None

        with open(sha256_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    hash_value = parts[0]
                    filename = parts[-1]  # Handle spaces in paths

                    if filename == f"{grid_item}.npy":
                        embedding_hash = hash_value
                    elif filename == f"{grid_item}_scales.npy":
                        scales_hash = hash_value

        # Validate required files and hashes
        embedding_path = os.path.join(grid_path, f"{grid_item}.npy")
        scales_path = os.path.join(grid_path, f"{grid_item}_scales.npy")

        embedding_exists = os.path.exists(embedding_path)
        scales_exists = os.path.exists(scales_path)

        # Skip directories with incomplete npy/scales files (one but not the other)
        if embedding_exists and not scales_exists:
            logger.warning(
                f"Skipping directory with incomplete files (missing scales): {grid_path}"
            )
            return None
        if scales_exists and not embedding_exists:
            logger.warning(
                f"Skipping directory with incomplete files (missing embedding): {grid_path}"
            )
            return None
        if not embedding_exists and not scales_exists:
            # Both missing - skip silently (probably not a tile directory)
            return None

        # Check for hashes in SHA256 file
        if embedding_hash is None:
            logger.warning(
                f"Skipping directory - no hash found for embedding in SHA256 file: {sha256_file}"
            )
            return None
        if scales_hash is None:
            logger.warning(
                f"Skipping directory - no hash found for scales in SHA256 file: {sha256_file}"
            )
            return None

        # Get file stats
        embedding_stat = os.stat(embedding_path)
        scales_stat = os.stat(scales_path)

        # Create TileInfo object
        tile_info = TileInfo(
            year=year,
            lat=lat,
            lon=lon,
            grid_name=grid_item,
            embedding_path=embedding_path,
            scales_path=scales_path,
            directory_path=grid_path,
            embedding_hash=embedding_hash,
            scales_hash=scales_hash,
            embedding_mtime=embedding_stat.st_mtime,
            scales_mtime=scales_stat.st_mtime,
            embedding_size=embedding_stat.st_size,
            scales_size=scales_stat.st_size,
        )

        return tile_info

    except Exception as e:
        # Return error info as a special tuple
        return ("ERROR", grid_item, str(e))


def iterate_tessera_tiles(
    base_dir: str,
    callback: Callable[[TileInfo], Any],
    progress_callback: Optional[Callable] = None,
) -> List[Any]:
    """
    Single-pass iterator through Tessera embedding filesystem structure.

    For each tile:
    1. Reads hashes from existing SHA256 file in grid directory
    2. Gets file stats (mtime, size) from filesystem
    3. Calls callback with complete TileInfo object
    4. Validates that both embedding and scales files exist

    Uses multiprocessing for parallel directory scanning.

    Args:
        base_dir: Base directory containing global_0.1_degree_representation
        callback: Function called for each tile with TileInfo object
        progress_callback: Optional progress reporting function(current, total, status)

    Returns:
        List of results from callback calls (None results are filtered out)

    Raises:
        FileNotFoundError: Missing SHA256 files or embedding/scales files
        ValueError: Corrupted directory structure or hash format
        OSError: Filesystem access errors
    """
    repr_dir = os.path.join(base_dir, "global_0.1_degree_representation")
    if not os.path.exists(repr_dir):
        raise FileNotFoundError(f"Embeddings directory not found: {repr_dir}")

    results = []
    processed_dirs = 0
    total_dirs = 0

    # First count total directories for progress (all grid directories, even empty ones)
    for year_item in os.listdir(repr_dir):
        year_path = os.path.join(repr_dir, year_item)
        if os.path.isdir(year_path) and year_item.isdigit() and len(year_item) == 4:
            for grid_item in os.listdir(year_path):
                grid_path = os.path.join(year_path, grid_item)
                if os.path.isdir(grid_path) and grid_item.startswith("grid_"):
                    total_dirs += 1

    if total_dirs == 0:
        # No grid directories at all
        return results  # Return empty list instead of raising error

    # Get number of CPU cores for parallel processing
    num_cores = multiprocessing.cpu_count()

    # Report parallelization to user if progress callback is available
    if progress_callback:
        progress_callback(
            0, total_dirs, f"Using {num_cores} CPU cores for parallel processing"
        )

    # Collect all grid directory tasks
    grid_tasks = []
    for year_item in os.listdir(repr_dir):
        year_path = os.path.join(repr_dir, year_item)
        if not (
            os.path.isdir(year_path) and year_item.isdigit() and len(year_item) == 4
        ):
            continue

        year = int(year_item)

        for grid_item in os.listdir(year_path):
            grid_path = os.path.join(year_path, grid_item)
            if os.path.isdir(grid_path) and grid_item.startswith("grid_"):
                grid_tasks.append((year, year_path, grid_item, base_dir))

    # Process grid directories in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_grid_directory, task): task for task in grid_tasks
        }

        # Process results as they complete
        for future in as_completed(futures):
            task = futures[future]
            year, year_path, grid_item, _ = task

            processed_dirs += 1

            # Update progress
            if progress_callback:
                progress_callback(processed_dirs, total_dirs, f"Processing {grid_item}")

            try:
                tile_info_or_error = future.result()

                # Check if this is an error result
                if (
                    isinstance(tile_info_or_error, tuple)
                    and len(tile_info_or_error) == 3
                ):
                    if tile_info_or_error[0] == "ERROR":
                        _, grid_name, error_msg = tile_info_or_error
                        grid_path = os.path.join(year_path, grid_name)
                        raise RuntimeError(f"Error processing {grid_path}: {error_msg}")

                # Skip None results (empty/skipped directories)
                if tile_info_or_error is None:
                    continue

                # Call callback and collect result
                result = callback(tile_info_or_error)
                if result is not None:
                    results.append(result)

            except Exception as e:
                # Stop on first error as requested
                if isinstance(e, RuntimeError):
                    raise
                else:
                    raise RuntimeError(
                        f"Unexpected error in parallel processing: {e}"
                    ) from e

    return results


def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def process_file(args):
    """Process a single file and return its relative path and hash."""
    file_path, base_dir, skip_checksum = args
    try:
        rel_path = os.path.relpath(file_path, base_dir)
        if skip_checksum:
            file_hash = ""
        else:
            file_hash = calculate_sha256(file_path)
        return rel_path, file_hash
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return None, None


def load_existing_registry(registry_path):
    """Load existing registry file into a dictionary."""
    registry = {}
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        registry[parts[0]] = parts[1]
    return registry


def find_npy_files_by_blocks(base_dir):
    """Find all .npy files and organize them by year and block."""
    files_by_year_and_block = defaultdict(lambda: defaultdict(list))

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".npy"):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, base_dir)

                # Extract year from path (assuming format ./YYYY/...)
                path_parts = rel_path.split(os.sep)
                if (
                    len(path_parts) > 0
                    and path_parts[0].isdigit()
                    and len(path_parts[0]) == 4
                ):
                    year = path_parts[0]

                    # Extract coordinates from the grid directory name
                    grid_dir = os.path.basename(os.path.dirname(file_path))
                    lon, lat = parse_grid_name(grid_dir)

                    if lon is not None and lat is not None:
                        block_lon, block_lat = block_from_world(lon, lat)
                        block_key = (block_lon, block_lat)
                        files_by_year_and_block[year][block_key].append(file_path)

    return files_by_year_and_block


def find_tiff_files_by_blocks(base_dir):
    """Find all .tiff files and organize them by block."""
    files_by_block = defaultdict(list)

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".tiff"):
                file_path = os.path.join(root, file)

                # Extract coordinates from the tiff filename (e.g., grid_-120.55_53.45.tiff)
                filename = os.path.basename(file_path)
                tiff_name = filename.replace(".tiff", "")
                lon, lat = parse_grid_name(tiff_name)

                if lon is not None and lat is not None:
                    block_lon, block_lat = block_from_world(lon, lat)
                    block_key = (block_lon, block_lat)
                    files_by_block[block_key].append(file_path)

    return files_by_block


def generate_master_registry(registry_dir):
    """Generate a master registry.txt file containing hashes of all registry files."""
    # This function is no longer used but kept for compatibility
    # The actual generation of registry.txt should be done separately
    pass


def create_landmasks_parquet_database(base_dir, output_path, console):
    """Create a Parquet database for landmasks by reading from SHA256SUM file.

    Args:
        base_dir: Base directory containing global_0.1_degree_tiff_all
        output_path: Output path for the Parquet file
        console: Rich console for output

    Returns:
        True on success, False on failure
    """
    console.print(
        Panel.fit(
            f"[bold blue]🗺️ Creating Landmasks Parquet Database[/bold blue]\n"
            f"📁 {base_dir}\n"
            f"📄 {output_path}",
            style="blue",
        )
    )

    sha256sum_file = os.path.join(base_dir, "SHA256SUM")
    if not os.path.exists(sha256sum_file):
        console.print(f"[red]SHA256SUM file not found:[/red] {sha256sum_file}")
        return False

    records = []

    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        console=console,
    ) as progress:
        read_task = progress.add_task(
            "Reading SHA256SUM file...", total=100, status="Starting..."
        )

        try:
            with open(sha256sum_file, "r") as f:
                lines = f.readlines()

            progress.update(read_task, completed=50, status="Parsing entries...")

            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        checksum = parts[0]
                        filename = parts[-1]

                        if filename.endswith(".tiff") or filename.endswith(".tif"):
                            if filename.startswith("grid_"):
                                try:
                                    # Remove 'grid_' prefix and '.tiff' suffix
                                    coords_str = (
                                        filename[5:]
                                        .replace(".tiff", "")
                                        .replace(".tif", "")
                                    )
                                    lon_str, lat_str = coords_str.split("_")
                                    lon = float(lon_str)
                                    lat = float(lat_str)

                                    # Get file size
                                    file_path = os.path.join(base_dir, filename)
                                    file_size = (
                                        os.path.getsize(file_path)
                                        if os.path.exists(file_path)
                                        else 0
                                    )

                                    records.append(
                                        {
                                            "lat": lat,
                                            "lon": lon,
                                            "hash": checksum,
                                            "file_size": file_size,
                                        }
                                    )
                                except (ValueError, IndexError):
                                    continue

            progress.update(read_task, completed=100, status="Complete")

        except Exception as e:
            console.print(f"[red]Error reading SHA256SUM file: {e}[/red]")
            return False

        if not records:
            console.print("[red]No landmask tiles found in SHA256SUM file[/red]")
            return False

        # Convert to GeoParquet
        parquet_task = progress.add_task(
            "Creating GeoParquet database...",
            total=100,
            status="Converting to DataFrame...",
        )

        progress.update(parquet_task, completed=25, status="Sorting records...")
        df = pd.DataFrame(records)
        df = df.sort_values(["lat", "lon"])

        # Add integer grid indices for robust cross-platform lookups
        df["lon_i"] = (df["lon"] * 100).round().astype(np.int32)
        df["lat_i"] = (df["lat"] * 100).round().astype(np.int32)

        progress.update(parquet_task, completed=50, status="Creating geometries...")
        # Convert to GeoDataFrame with Point geometries
        geometry = gpd.points_from_xy(df["lon"], df["lat"])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        progress.update(parquet_task, completed=75, status="Writing GeoParquet file...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first for atomic operation
        import tempfile

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=Path(output_path).parent,
                prefix=f".{Path(output_path).name}_tmp_",
                suffix=".parquet",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name

            gdf.to_parquet(temp_path, compression="zstd", index=False)
            os.rename(temp_path, output_path)

        except Exception:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        progress.update(parquet_task, completed=100, status="Complete")

    # Get file size and show results
    file_size = Path(output_path).stat().st_size

    # Summary table
    summary_table = Table(show_header=False, box=None)
    summary_table.add_row("📊 Records:", f"{len(records):,}")
    summary_table.add_row("💾 File size:", f"{file_size:,} bytes")
    summary_table.add_row(
        "🌍 Coordinates:",
        f"{len(gdf[['lat', 'lon']].drop_duplicates()):,} unique tiles",
    )
    summary_table.add_row("🗺️ Format:", "GeoParquet with Point geometries")

    console.print(
        Panel(
            summary_table,
            title="[bold green]✅ Landmasks GeoParquet Database Created[/bold green]",
            border_style="green",
        )
    )

    return True


def create_parquet_database_from_filesystem(base_dir, output_path, console):
    """Create a Parquet database by reading from existing SHA256 files.

    Fast implementation that reads hashes from SHA256 files instead of
    recalculating them, making database creation much faster.
    Uses temporary file for atomic writing to ensure cron-safe operation.

    Args:
        base_dir: Base directory containing global_0.1_degree_representation
        output_path: Output path for the Parquet file
        console: Rich console for output
    """
    # Show initial header
    console.print(
        Panel.fit(
            f"[bold blue]🗄️ Creating Embeddings Parquet Database[/bold blue]\n"
            f"📁 {base_dir}\n"
            f"📄 {output_path}",
            style="blue",
        )
    )

    # Show CPU core count for parallel processing
    num_cores = multiprocessing.cpu_count()
    console.print(
        f"[cyan]Using {num_cores} CPU cores for parallel tile scanning[/cyan]"
    )

    records = []

    def collect_tile_data(tile_info: TileInfo):
        """Callback to collect data for each tile."""
        return {
            "lat": tile_info.lat,
            "lon": tile_info.lon,
            "year": tile_info.year,
            "hash": tile_info.embedding_hash,  # From SHA256 file, no recalculation!
            "scales_hash": tile_info.scales_hash,  # Scales file hash
            "mtime": tile_info.embedding_mtime,
            "file_size": tile_info.embedding_size,
            "scales_size": tile_info.scales_size,  # Scales file size
        }

    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        console=console,
    ) as progress:
        # Single progress task for the iterator
        process_task = progress.add_task(
            "Reading tile metadata...", total=100, status="Starting..."
        )

        def progress_callback(current, total, status):
            progress.update(process_task, completed=current, total=total, status=status)

        try:
            # Use the fast iterator - reads SHA256 files, no hash calculation
            records = iterate_tessera_tiles(
                base_dir, collect_tile_data, progress_callback=progress_callback
            )

            progress.update(process_task, completed=100, status="Complete")

        except Exception as e:
            console.print(f"[red]Error iterating tiles: {e}[/red]")
            return False

        if not records:
            console.print(
                "[red]No tiles found. Make sure 'geotessera-registry hash' has been run first.[/red]"
            )
            return False

        # Convert to Parquet
        parquet_task = progress.add_task(
            "Creating Parquet database...",
            total=100,
            status="Converting to DataFrame...",
        )

        progress.update(parquet_task, completed=25, status="Sorting records...")
        df = pd.DataFrame(records)
        df = df.sort_values(["year", "lat", "lon"])

        progress.update(parquet_task, completed=40, status="Converting timestamps...")
        df["mtime"] = pd.to_datetime(df["mtime"], unit="s")

        # Add integer grid indices for robust cross-platform lookups
        df["lon_i"] = (df["lon"] * 100).round().astype(np.int32)
        df["lat_i"] = (df["lat"] * 100).round().astype(np.int32)

        progress.update(parquet_task, completed=55, status="Creating geometries...")
        # Convert to GeoDataFrame with Point geometries
        geometry = gpd.points_from_xy(df["lon"], df["lat"])
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

        progress.update(parquet_task, completed=75, status="Writing GeoParquet file...")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first for atomic operation (cron-safe)
        import tempfile

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=Path(output_path).parent,
                prefix=f".{Path(output_path).name}_tmp_",
                suffix=".parquet",
                delete=False,
            ) as temp_file:
                temp_path = temp_file.name

            gdf.to_parquet(temp_path, compression="zstd", index=False)
            os.rename(temp_path, output_path)

        except Exception:
            # Clean up temporary file on error
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        progress.update(parquet_task, completed=100, status="Complete")

    # Get file size and show results
    file_size = Path(output_path).stat().st_size

    # Summary table
    summary_table = Table(show_header=False, box=None)
    summary_table.add_row("📊 Records:", f"{len(records):,}")
    summary_table.add_row("💾 File size:", f"{file_size:,} bytes")
    summary_table.add_row("🗓️ Years:", ", ".join(map(str, sorted(gdf["year"].unique()))))
    summary_table.add_row(
        "🌍 Coordinates:",
        f"{len(gdf[['lat', 'lon']].drop_duplicates()):,} unique tiles",
    )
    summary_table.add_row("🗺️ Format:", "GeoParquet with Point geometries")

    console.print(
        Panel(
            summary_table,
            title="[bold green]✅ GeoParquet Database Created[/bold green]",
            border_style="green",
        )
    )

    return True


def check_command(args):
    """Check the integrity of the tessera filesystem structure."""
    console = Console()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        console.print(f"[red]Error: Directory {base_dir} does not exist[/red]")
        return 1

    verify_hashes = getattr(args, "verify_hashes", False)

    # Show header
    console.print(
        Panel.fit(
            f"[bold blue]🔍 Checking Tessera Structure[/bold blue]\n"
            f"📁 {base_dir}\n"
            f"🔐 Hash verification: {'enabled' if verify_hashes else 'disabled'}",
            style="blue",
        )
    )

    checked_tiles = 0

    def validate_tile(tile_info: TileInfo):
        """Callback to validate each tile."""
        nonlocal checked_tiles
        checked_tiles += 1

        # Check if files exist (already done by iterator, but good to be explicit)
        if not os.path.exists(tile_info.embedding_path):
            raise FileNotFoundError(f"Missing embedding: {tile_info.embedding_path}")
        if not os.path.exists(tile_info.scales_path):
            raise FileNotFoundError(f"Missing scales: {tile_info.scales_path}")

        # Verify hashes if requested
        if verify_hashes:
            actual_embedding_hash = calculate_sha256(tile_info.embedding_path)
            if actual_embedding_hash != tile_info.embedding_hash:
                raise ValueError(
                    f"Hash mismatch in {tile_info.embedding_path}: "
                    f"expected {tile_info.embedding_hash}, got {actual_embedding_hash}"
                )

            actual_scales_hash = calculate_sha256(tile_info.scales_path)
            if actual_scales_hash != tile_info.scales_hash:
                raise ValueError(
                    f"Hash mismatch in {tile_info.scales_path}: "
                    f"expected {tile_info.scales_hash}, got {actual_scales_hash}"
                )

        return "OK"

    # Run validation with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[status]}", justify="left"),
        console=console,
    ) as progress:
        check_task = progress.add_task(
            "Validating tiles...", total=100, status="Starting..."
        )

        def progress_callback(current, total, status):
            progress.update(check_task, completed=current, total=total, status=status)

        try:
            # Run validation
            iterate_tessera_tiles(
                base_dir, validate_tile, progress_callback=progress_callback
            )

            progress.update(check_task, completed=100, status="Complete")

        except Exception as e:
            console.print(f"[red]{emoji('❌ ')}Validation failed: {e}[/red]")
            return 1

    # Show results
    summary_table = Table(show_header=False, box=None)
    summary_table.add_row("🔍 Tiles checked:", f"{checked_tiles:,}")
    summary_table.add_row("✅ Status:", "All checks passed")
    if verify_hashes:
        summary_table.add_row("🔐 Hash verification:", "All hashes verified")
    else:
        summary_table.add_row("🔐 Hash verification:", "Skipped (use --verify-hashes)")

    console.print(
        Panel(
            summary_table,
            title="[bold green]✅ Tessera Structure Check Complete[/bold green]",
            border_style="green",
        )
    )

    return 0


def list_command(args):
    """List existing registry files in the specified directory."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        logger.error(f"Directory {base_dir} does not exist")
        return

    logger.info(f"Scanning for registry files in: {base_dir}")

    # Find all embeddings_*.txt and landmasks_*.txt files
    registry_files = []
    for file in os.listdir(base_dir):
        if (
            file.startswith("embeddings_") or file.startswith("landmasks_")
        ) and file.endswith(".txt"):
            registry_path = os.path.join(base_dir, file)
            # Count entries in the registry
            try:
                with open(registry_path, "r") as f:
                    entry_count = sum(
                        1 for line in f if line.strip() and not line.startswith("#")
                    )
                registry_files.append((file, entry_count))
            except Exception:
                registry_files.append((file, -1))

    if not registry_files:
        logger.warning("No registry files found")
        return

    # Sort by filename
    registry_files.sort()

    logger.info(f"\nFound {len(registry_files)} registry files:")
    for filename, count in registry_files:
        if count >= 0:
            logger.info(f"  - {filename}: {count:,} entries")
        else:
            logger.warning(f"  - {filename}: (error reading file)")

    # Check for master registry
    master_registry = os.path.join(base_dir, "registry.txt")
    if os.path.exists(master_registry):
        logger.info("\nMaster registry found: registry.txt")


def process_grid_checksum(args):
    """Process a single grid directory to generate SHA256 checksums.

    Only recalculates checksums if:
    - SHA256 file doesn't exist, OR
    - force=True, OR
    - Any .npy file has mtime newer than the SHA256 file
    """
    year_dir, grid_name, force, dry_run = args
    grid_dir = os.path.join(year_dir, grid_name)
    sha256_file = os.path.join(grid_dir, "SHA256")

    # Find all .npy files in this grid directory
    npy_files = [f for f in os.listdir(grid_dir) if f.endswith(".npy")]

    # If SHA256 file exists and force is not enabled, check mtimes
    if not force and os.path.exists(sha256_file):
        sha256_mtime = os.path.getmtime(sha256_file)

        # Check if any .npy file is newer than the SHA256 file
        needs_update = False
        newer_files = []
        for npy_file in npy_files:
            npy_path = os.path.join(grid_dir, npy_file)
            npy_mtime = os.path.getmtime(npy_path)
            if npy_mtime > sha256_mtime:
                needs_update = True
                newer_files.append((npy_file, npy_mtime - sha256_mtime))

        if not needs_update:
            # All files are older than SHA256 file, skip
            return (grid_name, len(npy_files), True, "skipped", None)
        elif dry_run:
            # In dry-run mode, report what would be updated
            return (grid_name, len(npy_files), True, "would_update", newer_files)

    elif not os.path.exists(sha256_file) and dry_run:
        # No SHA256 file exists - would need to create
        return (grid_name, len(npy_files), True, "would_create", None)

    if dry_run:
        # In dry-run mode, don't actually do anything
        return (grid_name, len(npy_files), True, "dry_run", None)

    if npy_files:
        try:
            # Change to grid directory and run sha256sum
            result = subprocess.run(
                ["sha256sum"] + sorted(npy_files),
                cwd=grid_dir,
                capture_output=True,
                text=True,
                check=True,
            )

            # Write output to SHA256 file
            with open(sha256_file, "w", encoding="utf-8") as f:
                f.write(result.stdout)

            return (grid_name, len(npy_files), True, None, None)
        except subprocess.CalledProcessError as e:
            return (grid_name, len(npy_files), False, f"CalledProcessError: {e}", None)
        except Exception as e:
            return (grid_name, len(npy_files), False, f"Exception: {e}", None)

    return (grid_name, 0, True, None, None)


def generate_embeddings_checksums(
    base_dir, force=False, dry_run=False, year_filter=None
):
    """Generate SHA256 checksums for .npy files in each embeddings subdirectory.

    Only recalculates checksums for grid directories where:
    - SHA256 file doesn't exist, OR
    - force=True, OR
    - Any .npy file has mtime newer than the SHA256 file

    This optimization makes subsequent runs much faster by only updating
    checksums when files have actually changed.

    Args:
        base_dir: Base directory containing year subdirectories
        force: Force regeneration even if SHA256 files are up to date
        dry_run: Don't actually update files, just report what would be done
        year_filter: Optional year to process (e.g. 2024), or None for all years
    """
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )

    # Print header info before starting progress display
    if dry_run:
        console.print("[yellow]DRY RUN MODE - no files will be modified[/yellow]")
    console.print("Generating SHA256 checksums for embeddings...")
    if force:
        console.print(
            "[yellow]Force mode enabled - regenerating all checksums[/yellow]"
        )
    else:
        console.print("Using smart update: only recalculating for modified files")

    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    console.print(f"Using [cyan]{num_cores}[/cyan] CPU cores for parallel processing")

    # Process each year directory
    year_dirs = []
    for item in os.listdir(base_dir):
        if item.isdigit() and len(item) == 4:  # Year directories
            # Apply year filter if specified
            if year_filter is not None and int(item) != year_filter:
                continue
            year_path = os.path.join(base_dir, item)
            if os.path.isdir(year_path):
                year_dirs.append(item)

    if not year_dirs:
        if year_filter:
            console.print(f"[red]No year directory found for {year_filter}[/red]")
        else:
            console.print("[red]No year directories found[/red]")
        return 1

    total_grids = 0
    processed_grids = 0
    skipped_total = 0
    would_update_total = 0
    would_create_total = 0
    errors = []

    # For dry-run, collect details
    update_details = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for year in sorted(year_dirs):
            year_dir = os.path.join(base_dir, year)

            # Find all grid directories
            grid_dirs = []
            for item in os.listdir(year_dir):
                if item.startswith("grid_"):
                    grid_path = os.path.join(year_dir, item)
                    if os.path.isdir(grid_path):
                        grid_dirs.append(item)

            total_grids += len(grid_dirs)

            # Prepare arguments for parallel processing
            grid_args = [
                (year_dir, grid_name, force, dry_run) for grid_name in sorted(grid_dirs)
            ]

            # Process grid directories in parallel
            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_grid_checksum, args): args
                    for args in grid_args
                }

                # Process results with progress bar
                skipped_grids = 0
                would_update_grids = 0
                would_create_grids = 0
                task = progress.add_task(f"Year {year}", total=len(grid_dirs))
                for future in as_completed(futures):
                    grid_name, num_files, success, error_msg, details = future.result()

                    if success:
                        if error_msg == "skipped":
                            skipped_grids += 1
                        elif error_msg == "would_update":
                            would_update_grids += 1
                            if (
                                details and len(update_details) < 20
                            ):  # Limit to first 20 examples
                                update_details.append((year, grid_name, details))
                        elif error_msg == "would_create":
                            would_create_grids += 1
                            if len(update_details) < 20:
                                update_details.append((year, grid_name, "no_sha256"))
                        elif num_files > 0:
                            processed_grids += 1
                    else:
                        errors.append(f"{grid_name}: {error_msg}")

                    progress.update(task, advance=1)

            # Store counters
            skipped_total += skipped_grids
            would_update_total += would_update_grids
            would_create_total += would_create_grids

    # Report any errors (after progress bars are done)
    if errors:
        console.print("\n[red]Errors encountered:[/red]")
        for error in errors[:10]:  # Show first 10 errors
            console.print(f"  [red]- {error}[/red]")
        if len(errors) > 10:
            console.print(f"  [dim]... and {len(errors) - 10} more errors[/dim]")

    # Print summary after progress bars
    console.print()  # Blank line
    if dry_run:
        console.print("=" * 60)
        console.print("[bold cyan]DRY RUN SUMMARY[/bold cyan]")
        console.print("=" * 60)
        console.print(f"Total directories scanned: [cyan]{total_grids:,}[/cyan]")
        console.print(f"  Would skip (up to date): [green]{skipped_total:,}[/green]")
        console.print(
            f"  Would update (files modified): [yellow]{would_update_total:,}[/yellow]"
        )
        console.print(
            f"  Would create (no SHA256): [yellow]{would_create_total:,}[/yellow]"
        )

        if update_details:
            console.print("\n[bold]Example directories that would be updated:[/bold]")
            for year, grid_name, details in update_details[:10]:
                if details == "no_sha256":
                    console.print(
                        f"  [yellow]{year}/{grid_name}[/yellow] - No SHA256 file exists"
                    )
                else:
                    console.print(
                        f"  [yellow]{year}/{grid_name}[/yellow] - {len(details)} files newer than SHA256:"
                    )
                    for npy_file, delta in details[:3]:  # Show first 3 files
                        console.print(f"    - {npy_file} [dim](+{delta:.1f}s)[/dim]")
        console.print("\n[green]No files were modified (dry-run mode)[/green]")
        return 0
    else:
        console.print(
            f"Processed [cyan]{processed_grids:,}[/cyan] / [cyan]{total_grids:,}[/cyan] grid directories"
        )
        if skipped_total > 0:
            console.print(
                f"Skipped [green]{skipped_total:,}[/green] directories (SHA256 files up to date)"
            )
        return 0 if processed_grids > 0 else 1


def process_tiff_chunk(args):
    """Process a chunk of TIFF files to generate SHA256 checksums."""
    base_dir, chunk, chunk_num = args
    temp_file = os.path.join(base_dir, f".SHA256SUM.tmp{chunk_num}")

    try:
        # Run sha256sum on this chunk
        result = subprocess.run(
            ["sha256sum"] + chunk,
            cwd=base_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        # Write to temporary file
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        return (chunk_num, len(chunk), True, None, temp_file)
    except subprocess.CalledProcessError as e:
        return (chunk_num, len(chunk), False, f"CalledProcessError: {e}", temp_file)
    except Exception as e:
        return (chunk_num, len(chunk), False, f"Exception: {e}", temp_file)


def generate_tiff_checksums(base_dir, force=False):
    """Generate SHA256 checksums for TIFF files using chunked parallel processing.

    Only recalculates checksums if:
    - SHA256SUM file doesn't exist, OR
    - force=True, OR
    - Any .tiff file has mtime newer than the SHA256SUM file
    """
    from rich.progress import (
        Progress,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )

    logger.info("Generating SHA256 checksums for TIFF files...")
    if force:
        logger.info("Force mode enabled - regenerating all checksums")
    else:
        logger.info(
            "Using smart update: only recalculating if files have been modified"
        )

    # Get number of CPU cores
    num_cores = multiprocessing.cpu_count()
    logger.info(f"Using {num_cores} CPU cores for parallel processing")

    # Find all .tiff files
    tiff_files = []
    for file in os.listdir(base_dir):
        if file.endswith(".tiff") or file.endswith(".tif"):
            tiff_files.append(file)

    if not tiff_files:
        logger.warning("No TIFF files found")
        return 1

    # Check if SHA256SUM already exists and force is not enabled
    sha256sum_file = os.path.join(base_dir, "SHA256SUM")
    if not force and os.path.exists(sha256sum_file):
        sha256sum_mtime = os.path.getmtime(sha256sum_file)

        # Check if any TIFF file is newer than the SHA256SUM file
        needs_update = False
        for tiff_file in tiff_files:
            tiff_path = os.path.join(base_dir, tiff_file)
            if os.path.getmtime(tiff_path) > sha256sum_mtime:
                needs_update = True
                logger.debug(f"Found updated file: {tiff_file}")
                break

        if not needs_update:
            logger.info(
                "SHA256SUM file is up to date (all TIFF files are older). Skipping."
            )
            logger.info("Use --force to regenerate anyway.")
            return 0

    # Sort files for consistent ordering
    tiff_files.sort()
    total_files = len(tiff_files)
    logger.info(f"Found {total_files} TIFF files")

    # Process in chunks to avoid command line length limits
    chunk_size = 1000  # Process 1000 files at a time

    # Prepare chunks for parallel processing
    chunks = []
    for i in range(0, total_files, chunk_size):
        chunk = tiff_files[i : i + chunk_size]
        chunk_num = i // chunk_size + 1
        chunks.append((base_dir, chunk, chunk_num))

    temp_files = []
    errors = []

    try:
        # Process chunks in parallel
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("Computing checksums", total=total_files)

            with ProcessPoolExecutor(max_workers=num_cores) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(process_tiff_chunk, args): args for args in chunks
                }

                # Process results with progress bar
                results = []
                for future in as_completed(futures):
                    chunk_num, chunk_len, success, error_msg, temp_file = (
                        future.result()
                    )

                    if success:
                        results.append((chunk_num, temp_file))
                    else:
                        errors.append(f"Chunk {chunk_num}: {error_msg}")

                    progress.update(task, advance=chunk_len)

            # Sort results by chunk number to maintain order
            results.sort(key=lambda x: x[0])
            temp_files = [temp_file for _, temp_file in results]

        if errors:
            logger.error("\nErrors encountered during processing:")
            for error in errors:
                logger.error(f"  - {error}")
            # Clean up any temporary files
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            return 1

        # Concatenate all temporary files into final SHA256SUM
        logger.info("Concatenating results...")
        with open(sha256sum_file, "w", encoding="utf-8") as outfile:
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    with open(temp_file, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())

        # Clean up temporary files
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        logger.info(f"Successfully generated checksums for {total_files} files")
        logger.info(f"Checksums written to: {sha256sum_file}")
        return 0

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        # Clean up any temporary files
        for _, _, _, _, temp_file in chunks:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        return 1


def scan_command(args):
    """Scan SHA256 checksum files and generate parquet databases."""
    console = Console()

    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        console.print(f"[red]Error: Directory {base_dir} does not exist[/red]")
        return 1

    console.print(
        Panel.fit(f"🔍 Scanning Registry Data\n📁 {base_dir}", style="bold blue")
    )

    # Determine output directory
    if hasattr(args, "registry_dir") and args.registry_dir:
        output_dir = os.path.abspath(args.registry_dir)
    else:
        output_dir = base_dir

    # Look for expected directories
    repr_dir = os.path.join(base_dir, "global_0.1_degree_representation")
    tiles_dir = os.path.join(base_dir, "global_0.1_degree_tiff_all")

    # Create embeddings Parquet database
    embeddings_parquet_path = os.path.join(output_dir, "registry.parquet")
    if os.path.exists(repr_dir):
        if not create_parquet_database_from_filesystem(
            base_dir, embeddings_parquet_path, console
        ):
            console.print("[red]Failed to create embeddings parquet database[/red]")
            return 1
    else:
        console.print(f"[red]Error: Embeddings directory not found: {repr_dir}[/red]")
        return 1

    # Create landmasks Parquet database
    landmasks_parquet_path = os.path.join(output_dir, "landmasks.parquet")
    if os.path.exists(tiles_dir):
        if not create_landmasks_parquet_database(
            tiles_dir, landmasks_parquet_path, console
        ):
            console.print(
                "[yellow]Warning: Failed to create landmasks parquet database[/yellow]"
            )
            # Don't return error, landmasks are optional
    else:
        console.print(
            f"[yellow]Warning: Landmasks directory not found: {tiles_dir}[/yellow]"
        )

    # Show final summary
    summary_lines = ["[green]✅ Registry Generation Complete[/green]\n"]
    summary_lines.append("📊 Generated outputs:")
    summary_lines.append("• Parquet databases:")
    summary_lines.append(f"  → {embeddings_parquet_path} (embeddings)")
    if os.path.exists(landmasks_parquet_path):
        summary_lines.append(f"  → {landmasks_parquet_path} (landmasks)")
    summary_lines.append(f"📁 Output directory: {output_dir}")

    console.print(Panel.fit("\n".join(summary_lines), style="green"))

    return 0


def hash_command(args):
    """Generate SHA256 checksums for embeddings and TIFF files."""
    base_dir = os.path.abspath(args.base_dir)
    if not os.path.exists(base_dir):
        console.print(f"[red]Directory {base_dir} does not exist[/red]")
        return 1

    force = getattr(args, "force", False)
    dry_run = getattr(args, "dry_run", False)
    year_filter = getattr(args, "year", None)

    # Check if this is an embeddings directory structure
    repr_dir = os.path.join(base_dir, "global_0.1_degree_representation")
    tiles_dir = os.path.join(base_dir, "global_0.1_degree_tiff_all")

    processed_any = False

    # Process embeddings if directory exists
    if os.path.exists(repr_dir):
        console.print(f"\n[bold]Processing embeddings directory:[/bold] {repr_dir}")
        if year_filter:
            console.print(f"[cyan]Filtering to year:[/cyan] {year_filter}")
        if (
            generate_embeddings_checksums(
                repr_dir, force=force, dry_run=dry_run, year_filter=year_filter
            )
            == 0
        ):
            processed_any = True

    # Process TIFF files if directory exists (no year filtering for TIFF files)
    if os.path.exists(tiles_dir) and year_filter is None:
        console.print(f"\n[bold]Processing TIFF directory:[/bold] {tiles_dir}")
        if generate_tiff_checksums(tiles_dir, force=force) == 0:
            processed_any = True
    elif os.path.exists(tiles_dir) and year_filter is not None:
        console.print(
            "[dim]Skipping TIFF processing (year filter only applies to embeddings)[/dim]"
        )

    if not processed_any:
        console.print("[red]No data directories found. Expected:[/red]")
        console.print(f"[red]  - {repr_dir}[/red]")
        console.print(f"[red]  - {tiles_dir}[/red]")
        return 1

    return 0


def analyze_registry_changes():
    """Analyze git changes in registry files and summarize by year."""
    try:
        # Get all changed registry files (staged, unstaged, and untracked)
        registry_files_changed = set()

        # Single git status call to get all changes
        result = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            # Parse git status output: XY filename (where XY is 2-char status)
            if len(line) >= 3:
                # Git status porcelain format: first 2 chars are status, then filename
                # Handle both "XY filename" and "XY  filename" formats
                filename = line[2:].lstrip()  # Skip status and any spaces

                if is_registry_file(filename):
                    registry_files_changed.add(filename)

        if not registry_files_changed:
            return {}, []

        # Analyze each changed file
        changes_by_year = defaultdict(lambda: {"added": 0, "modified": 0})
        registry_files_list = []

        for file_path in registry_files_changed:
            if not os.path.exists(file_path):
                continue

            year = extract_year_from_filename(file_path)
            is_new_file = is_untracked_file(file_path)

            if is_new_file:
                entries = count_entries_in_registry_file(file_path)
                changes_by_year[year]["added"] += entries
                registry_files_list.append(("A", file_path))
            else:
                added, removed = count_file_diff_entries(file_path)
                net_change = added - removed
                if net_change > 0:
                    changes_by_year[year]["added"] += net_change
                elif net_change < 0:
                    changes_by_year[year]["modified"] += abs(net_change)
                registry_files_list.append(("M", file_path))

        return changes_by_year, registry_files_list

    except subprocess.CalledProcessError as e:
        return None, f"Git command failed: {e}"
    except Exception as e:
        return None, f"Error analyzing changes: {e}"


def extract_year_from_filename(file_path):
    """Extract year from registry filename.

    Registry filenames follow these exact patterns:
    - embeddings_YYYY_lonX_latY.txt -> returns YYYY
    - landmasks_lonX_latY.txt -> returns None (no year)
    - registry_YYYY.txt -> returns YYYY
    - registry.txt -> returns None (master registry)
    """
    filename = os.path.basename(file_path)

    # embeddings_YYYY_lonX_latY.txt
    if filename.startswith("embeddings_"):
        parts = filename.split("_")
        if len(parts) >= 2 and parts[1].isdigit() and len(parts[1]) == 4:
            return int(parts[1])

    # registry_YYYY.txt
    elif filename.startswith("registry_") and filename.endswith(".txt"):
        year_str = filename[9:-4]  # Extract between 'registry_' and '.txt'
        if year_str.isdigit() and len(year_str) == 4:
            return int(year_str)

    # landmasks and master registry have no year
    return None


def is_registry_file(file_path):
    """Check if a file is a registry file we should analyze."""
    if not file_path.endswith(".txt"):
        return False

    filename = os.path.basename(file_path)
    return any(
        keyword in filename for keyword in ["registry", "embeddings", "landmasks"]
    )


def count_entries_in_registry_file(file_path):
    """Count the number of entries in a registry file."""
    if not os.path.exists(file_path):
        return 0

    try:
        with open(file_path, "r") as f:
            count = 0
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    count += 1
            return count
    except Exception:
        return 0


def is_untracked_file(file_path):
    """Check if a file is untracked (new)."""
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", file_path],
            capture_output=True,
            text=True,
        )
        return result.returncode != 0  # Non-zero means untracked
    except Exception:
        return False


def count_file_diff_entries(file_path):
    """Count added and removed entries in a modified registry file."""
    try:
        # Get the diff for this file
        result = subprocess.run(
            ["git", "diff", "HEAD", file_path],
            capture_output=True,
            text=True,
            check=True,
        )

        if not result.stdout.strip():
            return 0, 0  # No changes

        added = 0
        removed = 0

        for line in result.stdout.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                # Count non-comment, non-empty lines
                content = line[1:].strip()  # Remove the '+' prefix
                if content and not content.startswith("#"):
                    added += 1
            elif line.startswith("-") and not line.startswith("---"):
                # Count non-comment, non-empty lines
                content = line[1:].strip()  # Remove the '-' prefix
                if content and not content.startswith("#"):
                    removed += 1

        return added, removed

    except Exception:
        return 0, 0


def create_commit_message(changes_by_year, registry_files_changed):
    """Create a concise commit message from the changes analysis."""

    # Calculate totals
    total_added = sum(year_data["added"] for year_data in changes_by_year.values())
    total_modified = sum(
        year_data["modified"] for year_data in changes_by_year.values()
    )

    # Create summary line
    summary_parts = []
    if total_added > 0:
        summary_parts.append(f"{total_added} tiles added")
    if total_modified > 0:
        summary_parts.append(f"{total_modified} tiles modified")

    summary = (
        f"Update registry: {', '.join(summary_parts)}"
        if summary_parts
        else "Update registry files"
    )

    # Create concise message with year breakdown
    message_parts = [summary]

    if (
        changes_by_year and len(changes_by_year) > 1
    ):  # Only show breakdown if multiple years
        message_parts.append("")
        for year in sorted(y for y in changes_by_year.keys() if y is not None):
            year_data = changes_by_year[year]
            changes = []
            if year_data["added"] > 0:
                changes.append(f"+{year_data['added']}")
            if year_data["modified"] > 0:
                changes.append(f"~{year_data['modified']}")
            if changes:
                message_parts.append(f"{year}: {', '.join(changes)}")

    return "\n".join(message_parts)


def commit_command(args):
    """Analyze registry changes and create a commit with summary."""
    console = Console()

    # Check if we're in a git repository
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"], capture_output=True, check=True
        )
    except subprocess.CalledProcessError:
        console.print("[red]Error: Not in a git repository[/red]")
        console.print("[dim]Run this command from within a git repository[/dim]")
        return 1

    console.print(Panel.fit("📊 Analyzing Registry Changes", style="cyan"))

    # Check if git is properly configured
    try:
        subprocess.run(["git", "config", "user.name"], capture_output=True, check=True)
        subprocess.run(["git", "config", "user.email"], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        console.print(
            "[red]Error: Git user.name and user.email must be configured[/red]"
        )
        console.print(
            "[dim]Run: git config user.name 'Your Name' && git config user.email 'your@email.com'[/dim]"
        )
        return 1

    # Analyze changes
    changes_by_year, registry_files_changed = analyze_registry_changes()

    if changes_by_year is None:
        console.print(f"[red]Error analyzing changes: {registry_files_changed}[/red]")
        return 1

    if not registry_files_changed:
        console.print("[yellow]No registry file changes detected[/yellow]")
        console.print(
            "[dim]Only .txt files with 'registry', 'embeddings', or 'landmasks' in the name are analyzed[/dim]"
        )
        return 0

    # Display summary
    console.print(
        f"[green]Found {len(registry_files_changed)} registry files with changes[/green]"
    )

    # Display file-by-file breakdown
    console.print("\n[blue]Changed files:[/blue]")
    for status, file_path in registry_files_changed[:10]:  # Show first 10
        status_str = {
            "A": "[green]Added[/green]",
            "M": "[yellow]Modified[/yellow]",
            "D": "[red]Deleted[/red]",
        }.get(status, status)
        console.print(f"  {status_str}: {file_path}")

    if len(registry_files_changed) > 10:
        console.print(
            f"  [dim]... and {len(registry_files_changed) - 10} more files[/dim]"
        )

    if changes_by_year:
        console.print("\n[blue]Summary by year:[/blue]")
        total_added = 0
        total_modified = 0

        for year in sorted(changes_by_year.keys(), key=lambda x: (x is None, x)):
            year_str = str(year) if year else "unknown"
            year_data = changes_by_year[year]

            change_parts = []
            if year_data["added"] > 0:
                change_parts.append(f"[green]+{year_data['added']} tiles[/green]")
                total_added += year_data["added"]
            if year_data["modified"] > 0:
                change_parts.append(
                    f"[yellow]~{year_data['modified']} modified[/yellow]"
                )
                total_modified += year_data["modified"]

            if change_parts:
                console.print(f"  {year_str}: {', '.join(change_parts)}")
        console.print(
            f"\n[bold]Total: [green]+{total_added} added[/green]"
            + (
                f" / [yellow]~{total_modified} modified[/yellow]"
                if total_modified > 0
                else ""
            )
            + " tiles[/bold]"
        )

    # Validate we have something to commit
    if all(
        year_data["added"] == 0 and year_data["modified"] == 0
        for year_data in changes_by_year.values()
    ):
        console.print(
            "[yellow]Warning: No tile changes detected in registry files[/yellow]"
        )
        console.print(
            "[dim]Files may have been reformatted without content changes[/dim]"
        )

    # Stage registry files
    console.print("\n[blue]Staging registry files...[/blue]")
    staged_files = []
    failed_files = []

    for status, file_path in registry_files_changed:
        if os.path.exists(file_path):  # Only add files that exist
            try:
                subprocess.run(
                    ["git", "add", file_path], check=True, capture_output=True
                )
                staged_files.append(file_path)
            except subprocess.CalledProcessError as e:
                failed_files.append((file_path, str(e)))

    if staged_files:
        console.print(
            f"[green]{emoji('✓ ')}Staged {len(staged_files)} files successfully[/green]"
        )

    if failed_files:
        console.print(
            f"[yellow]{emoji('⚠ ')}Failed to stage {len(failed_files)} files:[/yellow]"
        )
        for file_path, error in failed_files[:3]:  # Show first 3 failures
            console.print(f"  {file_path}: {error}")
        if len(failed_files) > 3:
            console.print(f"  [dim]... and {len(failed_files) - 3} more failures[/dim]")

    # Check if there's anything staged
    try:
        result = subprocess.run(
            ["git", "diff", "--staged", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        if not result.stdout.strip():
            console.print("[yellow]No files staged for commit[/yellow]")
            return 1
    except subprocess.CalledProcessError:
        console.print("[red]Error checking staged files[/red]")
        return 1

    # Create commit message
    commit_message = create_commit_message(changes_by_year, registry_files_changed)

    console.print("\n[blue]Commit message:[/blue]")
    console.print(Panel(commit_message, style="dim"))

    # Create the commit
    console.print("[blue]Creating commit...[/blue]")
    try:
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            capture_output=True,
            text=True,
            check=True,
        )

        console.print(f"[green]{emoji('✓ ')}Commit created successfully[/green]")

        # Show the commit hash and stats
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        commit_hash = commit_result.stdout.strip()[:8]
        console.print(f"[cyan]Commit: {commit_hash}[/cyan]")

        # Show commit stats
        if result.stdout.strip():
            console.print(f"[dim]{result.stdout.strip()}[/dim]")

        return 0

    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error creating commit: {e}[/red]")
        if e.stderr:
            console.print(
                f"[dim]Git error: {e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}[/dim]"
            )
        return 1


def export_manifests_command(args):
    """Convert Parquet registry files to Pooch-format text manifests.

    This command reads registry.parquet and landmasks.parquet files and exports
    them to the old block-based text registry format for backwards compatibility.
    Useful for maintaining the tessera-manifests repository.
    """
    console = Console()

    # Resolve input directory
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory does not exist: {input_dir}[/red]")
        return 1

    # Find Parquet files
    registry_parquet = input_dir / "registry.parquet"
    landmasks_parquet = input_dir / "landmasks.parquet"

    if not registry_parquet.exists():
        console.print(f"[red]Error: registry.parquet not found in {input_dir}[/red]")
        return 1

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = input_dir / "registry"

    output_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        Panel.fit(
            f"[bold blue]📦 Converting Parquet to Text Manifests[/bold blue]\n"
            f"📁 Input: {input_dir}\n"
            f"📁 Output: {output_dir}",
            style="blue",
        )
    )

    total_files_written = 0

    # ========== Process Embeddings ==========
    console.print("\n[cyan]Processing embeddings registry...[/cyan]")

    try:
        df = pd.read_parquet(registry_parquet)
        console.print(f"  Loaded [green]{len(df):,}[/green] embedding records")

        # Verify required columns
        required_cols = ["lat", "lon", "year", "hash", "scales_hash"]
        missing = set(required_cols) - set(df.columns)
        if missing:
            console.print(
                f"[red]Error: Missing required columns in registry.parquet: {missing}[/red]"
            )
            console.print(f"[yellow]Available columns: {df.columns.tolist()}[/yellow]")
            console.print(
                "[yellow]Hint: Regenerate registry.parquet with latest geotessera-registry scan[/yellow]"
            )
            return 1

        # Group by year and block
        files_by_year_and_block = defaultdict(lambda: defaultdict(list))

        for _, row in df.iterrows():
            lon, lat, year = row["lon"], row["lat"], row["year"]
            embedding_hash = row["hash"]
            scales_hash = row["scales_hash"]

            # Calculate block coordinates
            block_lon, block_lat = block_from_world(lon, lat)

            # Construct file paths (matching the structure in data directories)
            grid_name = f"grid_{lon:.2f}_{lat:.2f}"
            embedding_path = f"{year}/{grid_name}/{grid_name}.npy"
            scales_path = f"{year}/{grid_name}/{grid_name}_scales.npy"

            # Add both files to the block with their respective hashes
            files_by_year_and_block[year][(block_lon, block_lat)].append(
                (embedding_path, embedding_hash)
            )
            files_by_year_and_block[year][(block_lon, block_lat)].append(
                (scales_path, scales_hash)
            )

        # Write embeddings registry files
        embeddings_dir = output_dir / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        for year in sorted(files_by_year_and_block.keys()):
            blocks = files_by_year_and_block[year]
            console.print(f"  Year {year}: {len(blocks)} blocks")

            for (block_lon, block_lat), entries in sorted(blocks.items()):
                registry_filename = block_to_embeddings_registry_filename(
                    str(year), block_lon, block_lat
                )
                registry_file = embeddings_dir / registry_filename

                # Write file atomically
                import tempfile

                temp_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        dir=embeddings_dir,
                        prefix=f".{registry_filename}_tmp_",
                        suffix=".txt",
                        delete=False,
                    ) as temp_file:
                        temp_path = temp_file.name
                        for rel_path, checksum in sorted(entries):
                            temp_file.write(f"{rel_path} {checksum}\n")

                    os.rename(temp_path, registry_file)
                    temp_path = None
                    total_files_written += 1

                except Exception:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
                    raise

                console.print(
                    f"    Block ({block_lon:4d}, {block_lat:4d}): {len(entries):4d} files → {registry_filename}"
                )

        console.print(
            f"[green]{emoji('✓ ')}Wrote {total_files_written} embeddings registry files[/green]"
        )

    except Exception as e:
        console.print(f"[red]Error processing embeddings: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1

    # ========== Process Landmasks ==========
    if landmasks_parquet.exists():
        console.print("\n[cyan]Processing landmasks registry...[/cyan]")

        try:
            df = pd.read_parquet(landmasks_parquet)
            console.print(f"  Loaded [green]{len(df):,}[/green] landmask records")

            # Verify required columns
            required_cols = ["lat", "lon", "hash"]
            missing = set(required_cols) - set(df.columns)
            if missing:
                console.print(
                    f"[yellow]Warning: Missing columns in landmasks.parquet: {missing}[/yellow]"
                )
            else:
                # Group by block
                files_by_block = defaultdict(list)

                for _, row in df.iterrows():
                    lon, lat = row["lon"], row["lat"]
                    sha256 = row["hash"]

                    # Calculate block coordinates
                    block_lon, block_lat = block_from_world(lon, lat)

                    # Construct file path
                    filename = f"grid_{lon:.2f}_{lat:.2f}.tiff"

                    files_by_block[(block_lon, block_lat)].append((filename, sha256))

                # Write landmask registry files
                landmasks_dir = output_dir / "landmasks"
                landmasks_dir.mkdir(parents=True, exist_ok=True)

                landmask_files_written = 0
                console.print(f"  Writing {len(files_by_block)} landmask blocks")

                for (block_lon, block_lat), entries in sorted(files_by_block.items()):
                    registry_filename = block_to_landmasks_registry_filename(
                        block_lon, block_lat
                    )
                    registry_file = landmasks_dir / registry_filename

                    # Write file atomically
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            mode="w",
                            dir=landmasks_dir,
                            prefix=f".{registry_filename}_tmp_",
                            suffix=".txt",
                            delete=False,
                        ) as temp_file:
                            temp_path = temp_file.name
                            for rel_path, checksum in sorted(entries):
                                temp_file.write(f"{rel_path} {checksum}\n")

                        os.rename(temp_path, registry_file)
                        temp_path = None
                        landmask_files_written += 1

                    except Exception:
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)
                        raise

                    console.print(
                        f"    Block ({block_lon:4d}, {block_lat:4d}): {len(entries):4d} files → {registry_filename}"
                    )

                console.print(
                    f"[green]{emoji('✓ ')}Wrote {landmask_files_written} landmask registry files[/green]"
                )
                total_files_written += landmask_files_written

        except Exception as e:
            console.print(f"[yellow]Warning: Error processing landmasks: {e}[/yellow]")
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
    else:
        console.print(
            f"[yellow]Landmasks parquet not found, skipping: {landmasks_parquet}[/yellow]"
        )

    # Summary
    console.print(
        Panel.fit(
            f"[green]✅ Export Complete[/green]\n"
            f"📊 Total registry files written: {total_files_written}\n"
            f"📁 Output directory: {output_dir}",
            style="green",
        )
    )

    return 0


def file_scan_command(args):
    """Recursively scan year directories for embedding tiles and generate a parquet inventory.

    This command scans an input directory expecting year subdirectories (e.g., 2024/, 2023/).
    Within each year directory, it recursively searches for embedding tiles (identified by
    grid*.npy and grid*_scales.npy files). It extracts coordinates from filenames and records
    modification times for both files, along with the year.

    Useful for finding potential duplicate embeddings across machines.
    """
    import re
    from datetime import datetime

    console = Console()

    # Resolve input directory
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        console.print(f"[red]Error: Input directory does not exist: {input_dir}[/red]")
        return 1

    # Determine output file
    if args.output:
        output_file = Path(args.output).resolve()
    else:
        output_file = input_dir / "embedding_inventory.parquet"

    console.print(
        Panel.fit(
            f"[bold blue]🔍 Scanning for Embedding Tiles[/bold blue]\n"
            f"📁 Input: {input_dir}\n"
            f"📄 Output: {output_file}",
            style="blue",
        )
    )

    # Pattern to match grid files
    grid_pattern = re.compile(r"grid_(-?\d+\.\d+)_(-?\d+\.\d+)\.npy$")

    # Collect data
    records = []
    processed_dirs = set()
    years_found = set()

    console.print("\n[cyan]Scanning for year directories...[/cyan]")

    # First, identify year directories (subdirectories that are numeric years)
    year_dirs = []
    try:
        for item in input_dir.iterdir():
            if item.is_dir():
                # Check if directory name is a valid year (4 digits)
                if re.match(r"^\d{4}$", item.name):
                    year = int(item.name)
                    year_dirs.append((year, item))
                    console.print(f"  Found year directory: [green]{item.name}[/green]")
    except Exception as e:
        console.print(f"[red]Error scanning input directory: {e}[/red]")
        return 1

    if not year_dirs:
        console.print(
            "[yellow]No year directories found! Expected directories like 2024/, 2023/, etc.[/yellow]"
        )
        return 1

    console.print(
        f"\n[cyan]Scanning {len(year_dirs)} year directories for embedding tiles...[/cyan]"
    )

    # Walk each year directory
    for year, year_dir in sorted(year_dirs):
        console.print(f"\n[dim]Processing year {year}...[/dim]")

        for root, dirs, files in os.walk(year_dir):
            # Look for grid*.npy files in this directory
            grid_files = [f for f in files if grid_pattern.match(f)]

            if not grid_files:
                continue

            # Process each grid file found
            for grid_file in grid_files:
                match = grid_pattern.match(grid_file)
                if not match:
                    continue

                lon = float(match.group(1))
                lat = float(match.group(2))

                # Construct paths
                grid_name = f"grid_{lon:.2f}_{lat:.2f}"
                grid_path = os.path.join(root, f"{grid_name}.npy")
                scales_path = os.path.join(root, f"{grid_name}_scales.npy")

                # Check if both files exist
                if not os.path.exists(grid_path):
                    console.print(f"[yellow]Warning: Missing {grid_path}[/yellow]")
                    continue

                if not os.path.exists(scales_path):
                    console.print(
                        f"[yellow]Warning: Missing scales file for {grid_path}[/yellow]"
                    )
                    continue

                # Get modification times and full real paths
                try:
                    grid_stat = os.stat(grid_path)
                    scales_stat = os.stat(scales_path)

                    grid_mtime = datetime.fromtimestamp(grid_stat.st_mtime)
                    scales_mtime = datetime.fromtimestamp(scales_stat.st_mtime)
                    grid_size = grid_stat.st_size
                    scales_size = scales_stat.st_size

                    # Get full real paths
                    grid_realpath = os.path.realpath(grid_path)
                    scales_realpath = os.path.realpath(scales_path)

                    # Record the information
                    records.append(
                        {
                            "year": year,
                            "lon": lon,
                            "lat": lat,
                            "directory": root,
                            "grid_path": grid_realpath,
                            "scales_path": scales_realpath,
                            "grid_mtime": grid_mtime,
                            "scales_mtime": scales_mtime,
                            "grid_size": grid_size,
                            "scales_size": scales_size,
                        }
                    )

                    # Track unique directories processed
                    processed_dirs.add(root)
                    years_found.add(year)

                except Exception as e:
                    console.print(f"[red]Error processing {grid_path}: {e}[/red]")
                    continue

    if not records:
        console.print("[yellow]No embedding tiles found![/yellow]")
        return 1

    # Create DataFrame
    console.print(
        f"\n[cyan]Creating parquet file with {len(records):,} tiles...[/cyan]"
    )
    df = pd.DataFrame(records)

    # Sort by year, lon, lat for easier analysis
    df = df.sort_values(["year", "lon", "lat"])

    # Add integer grid indices for robust cross-platform lookups
    df["lon_i"] = (df["lon"] * 100).round().astype(np.int32)
    df["lat_i"] = (df["lat"] * 100).round().astype(np.int32)

    # Save to parquet
    try:
        # Create output directory if it doesn't exist
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(output_file, index=False)

        console.print(
            Panel.fit(
                f"[green]✅ Scan Complete[/green]\n"
                f"📊 Tiles found: {len(records):,}\n"
                f"📅 Years: {', '.join(str(y) for y in sorted(years_found))}\n"
                f"📁 Unique directories: {len(processed_dirs):,}\n"
                f"📄 Output: {output_file}",
                style="green",
            )
        )

        # Show sample of data
        console.print("\n[cyan]Sample of collected data:[/cyan]")
        table = Table(show_header=True)
        table.add_column("Year", style="magenta")
        table.add_column("Lon", style="cyan")
        table.add_column("Lat", style="cyan")
        table.add_column("Grid mtime", style="yellow")
        table.add_column("Scales mtime", style="yellow")
        table.add_column("Grid path", style="dim")

        for _, row in df.head(5).iterrows():
            table.add_row(
                str(row["year"]),
                f"{row['lon']:.2f}",
                f"{row['lat']:.2f}",
                row["grid_mtime"].strftime("%Y-%m-%d %H:%M:%S"),
                row["scales_mtime"].strftime("%Y-%m-%d %H:%M:%S"),
                str(row["grid_path"])[:50] + "..."
                if len(str(row["grid_path"])) > 50
                else str(row["grid_path"]),
            )

        console.print(table)

        return 0

    except Exception as e:
        console.print(f"[red]Error writing parquet file: {e}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1


def file_check_command(args):
    """Check multiple inventory parquet files for duplicate year/lon/lat coordinates.

    This command loads multiple parquet files (generated by file-scan) and identifies
    any year/lon/lat coordinates that appear in multiple files or multiple times within
    the same file. This is useful for finding duplicate embeddings across machines.
    """
    from collections import defaultdict

    console = Console()

    # Resolve input files
    parquet_files = [Path(f).resolve() for f in args.parquet_files]

    # Validate that all files exist
    missing_files = [f for f in parquet_files if not f.exists()]
    if missing_files:
        console.print("[red]Error: The following parquet files do not exist:[/red]")
        for f in missing_files:
            console.print(f"  - {f}")
        return 1

    console.print(
        Panel.fit(
            f"[bold blue]🔍 Checking for Duplicate Coordinates[/bold blue]\n"
            f"📊 Files to check: {len(parquet_files)}",
            style="blue",
        )
    )

    # Track coordinates and their sources
    # Key: (year, lon, lat), Value: list of location info dicts
    coord_locations = defaultdict(list)

    # Load each parquet file
    for parquet_file in parquet_files:
        console.print(f"\n[cyan]Loading {parquet_file.name}...[/cyan]")

        try:
            df = pd.read_parquet(parquet_file)

            # Verify required columns
            required_cols = ["year", "lon", "lat", "directory"]
            missing = set(required_cols) - set(df.columns)
            if missing:
                console.print(
                    f"[yellow]Warning: Missing required columns in {parquet_file.name}: {missing}[/yellow]"
                )
                console.print(
                    f"[yellow]Available columns: {df.columns.tolist()}[/yellow]"
                )
                continue

            console.print(f"  Found [green]{len(df):,}[/green] tiles")

            # Record each coordinate and its location
            for _, row in df.iterrows():
                year = row["year"]
                lon, lat = row["lon"], row["lat"]
                directory = row["directory"]
                grid_path = row.get("grid_path", None)
                grid_mtime = row.get("grid_mtime", None)
                scales_mtime = row.get("scales_mtime", None)

                coord_locations[(year, lon, lat)].append(
                    {
                        "parquet_file": parquet_file.name,
                        "directory": directory,
                        "grid_path": grid_path,
                        "grid_mtime": grid_mtime,
                        "scales_mtime": scales_mtime,
                    }
                )

        except Exception as e:
            console.print(f"[red]Error loading {parquet_file.name}: {e}[/red]")
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            continue

    # Find duplicates (coordinates that appear more than once)
    duplicates = {
        coord: locations
        for coord, locations in coord_locations.items()
        if len(locations) > 1
    }

    if not duplicates:
        console.print(
            Panel.fit(
                "[green]✅ No duplicate coordinates found![/green]\n"
                f"📊 Total unique coordinates: {len(coord_locations):,}",
                style="green",
            )
        )
        return 0

    # Display duplicates
    console.print(
        Panel.fit(
            f"[yellow]{emoji('⚠️  ')}Found {len(duplicates):,} duplicate coordinates[/yellow]\n"
            f"{emoji('📊 ')}Total unique coordinates: {len(coord_locations):,}",
            style="yellow",
        )
    )

    console.print("\n[bold]Duplicate Coordinates:[/bold]\n")

    # Sort duplicates by coordinate for consistent output
    for (year, lon, lat), locations in sorted(duplicates.items()):
        console.print(
            f"[bold cyan]Year {year}, Coordinate: ({lon:.2f}, {lat:.2f})[/bold cyan]"
        )
        console.print(f"  Found in [yellow]{len(locations)}[/yellow] locations:")

        for i, loc in enumerate(locations, 1):
            console.print(
                f"\n  [dim]{i}.[/dim] Parquet: [green]{loc['parquet_file']}[/green]"
            )
            console.print(f"     Directory: {loc['directory']}")
            if loc["grid_path"]:
                console.print(f"     Grid path: {loc['grid_path']}")
            if loc["grid_mtime"]:
                console.print(f"     Grid mtime: {loc['grid_mtime']}")
            if loc["scales_mtime"]:
                console.print(f"     Scales mtime: {loc['scales_mtime']}")

        console.print()

    # Summary statistics
    console.print("\n[bold]Summary by parquet file:[/bold]")
    file_dup_counts = defaultdict(int)
    for locations in duplicates.values():
        for loc in locations:
            file_dup_counts[loc["parquet_file"]] += 1

    table = Table(show_header=True)
    table.add_column("Parquet File", style="cyan")
    table.add_column("Duplicate Entries", style="yellow", justify="right")

    for parquet_file in sorted(file_dup_counts.keys()):
        table.add_row(parquet_file, f"{file_dup_counts[parquet_file]:,}")

    console.print(table)

    return 0


def main():
    """Main entry point for the geotessera-registry CLI tool."""
    # Configure logging with rich handler
    # Disable rich formatting in dumb terminals (use Rich Console's built-in detection)
    use_rich = console.is_terminal
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(
                rich_tracebacks=True, show_time=False, show_path=False, console=console
            )
        ]
        if use_rich
        else [logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(
        description="GeoTessera Registry Management Tool - Generate and maintain Pooch registry files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List existing registry files
  geotessera-registry list /path/to/data
  
  # Generate SHA256 checksums for embeddings and TIFF files
  geotessera-registry hash /path/to/v1

  # This will:
  # - Create SHA256 files in each grid subdirectory under global_0.1_degree_representation/YYYY/
  # - Create SHA256SUM file in global_0.1_degree_tiff_all/ using chunked processing
  # - Skip directories that already have SHA256 files (use --force to regenerate)

  # Dry run: see what would be recalculated without modifying files
  geotessera-registry hash /path/to/v1 --dry-run

  # Check a specific year only
  geotessera-registry hash /path/to/v1 --dry-run --year 2024

  # Force regeneration of all checksums
  geotessera-registry hash /path/to/v1 --force
  
  # Scan existing SHA256 checksum files and generate both parquet database and registry files
  geotessera-registry scan /path/to/v1
  
  # This will:
  # - Create a Parquet database with all tile metadata (lat/lon/year/hash/mtime)
  # - Read SHA256 files from grid subdirectories and generate block-based registry files
  # - Read SHA256SUM file from TIFF directory and generate landmask registry files
  # - Use atomic file writing (temp files) for cron-safe operation
  
  # Check filesystem structure integrity
  geotessera-registry check /path/to/v1
  
  # This will:
  # - Validate all directory structures are correct
  # - Check that all SHA256 files exist
  # - Verify that embedding and scales files exist
  # - Optionally verify SHA256 hashes (use --verify-hashes)
  
  # Analyze registry changes and create a git commit with detailed summary
  geotessera-registry commit

  # This will:
  # - Analyze git changes in registry files
  # - Summarize changes by year (tiles added/removed/modified)
  # - Stage registry files and create a commit with detailed message

  # Export Parquet registry to text manifests for backwards compatibility
  geotessera-registry export-manifests /path/to/v1

  # This will:
  # - Read registry.parquet and landmasks.parquet
  # - Generate block-based text registry files in registry/embeddings/ and registry/landmasks/
  # - Create separate entries for .npy and _scales.npy with their respective hashes
  # - Useful for maintaining the tessera-manifests repository

  # Export to custom output directory
  geotessera-registry export-manifests /path/to/v1 --output-dir ~/src/git/ucam-eo/tessera-manifests

  # Scan year directories for embedding tiles and create an inventory
  geotessera-registry file-scan /path/to/embeddings

  # This will:
  # - Scan for year subdirectories (e.g., 2024/, 2023/) in the input directory
  # - Recursively scan each year directory for grid*.npy files
  # - Extract lon/lat coordinates from filenames
  # - Record year, modification times for both grid*.npy and *_scales.npy files
  # - Record full real paths for embedding files
  # - Generate a parquet file with: year, lon, lat, directory, grid_path, scales_path, grid_mtime, scales_mtime, file sizes
  # - Useful for finding potential duplicate embeddings across machines

  # Specify custom output path
  geotessera-registry file-scan /path/to/embeddings --output /path/to/inventory.parquet

  # Check multiple inventory files for duplicate coordinates (year/lon/lat)
  geotessera-registry file-check machine1_inventory.parquet machine2_inventory.parquet machine3_inventory.parquet

  # This will:
  # - Load all specified parquet files (generated by file-scan)
  # - Find any year/lon/lat coordinates that appear in multiple files or multiple times
  # - Display duplicates with their year, source files, full paths, directories, and modification times
  # - Show summary statistics by parquet file
  # - Useful for identifying duplicate embeddings across generation machines

This tool is intended for GeoTessera data maintainers to generate the registry
files that are distributed with the package. End users typically don't need
to use this tool.

Note: This tool creates block-based registries for efficient lazy loading:
  - Embeddings: Organized into 5x5 degree blocks (embeddings_YYYY_lonX_latY.txt)
  - Landmasks: Organized into 5x5 degree blocks (landmasks_lonX_latY.txt)
  - Each block contains ~2,500 tiles instead of one massive registry
  - Registry files are created in the registry/ subdirectory

Directory Structure:
  The commands expect to find these subdirectories:
  - global_0.1_degree_representation/  (contains .npy files organized by year)
  - global_0.1_degree_tiff_all/        (contains .tiff files in flat structure)
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List existing registry files")
    list_parser.add_argument(
        "base_dir", help="Base directory to scan for registry files"
    )
    list_parser.set_defaults(func=list_command)

    # Hash command
    hash_parser = subparsers.add_parser(
        "hash", help="Generate SHA256 checksums for embeddings and TIFF files"
    )
    hash_parser.add_argument(
        "base_dir",
        help="Base directory containing global_0.1_degree_representation and/or global_0.1_degree_tiff_all subdirectories",
    )
    hash_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of all checksums, even if SHA256 files already exist",
    )
    hash_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be recalculated without actually modifying files",
    )
    hash_parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Only process a specific year (e.g., 2024). Applies to embeddings only.",
    )
    hash_parser.set_defaults(func=hash_command)

    # Scan command
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan existing SHA256 checksum files and generate both parquet database and registry files",
    )
    scan_parser.add_argument(
        "base_dir",
        help="Base directory containing global_0.1_degree_representation and/or global_0.1_degree_tiff_all subdirectories with checksum files",
    )
    scan_parser.add_argument(
        "--registry-dir",
        type=str,
        default=None,
        help="Output directory for registry and parquet files (default: same as base_dir)",
    )
    scan_parser.set_defaults(func=scan_command)

    # Check command
    check_parser = subparsers.add_parser(
        "check",
        help="Check the integrity of tessera filesystem structure and validate files",
    )
    check_parser.add_argument(
        "base_dir",
        help="Base directory containing global_0.1_degree_representation subdirectory",
    )
    check_parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Recalculate and verify SHA256 hashes (slower but thorough)",
    )
    check_parser.set_defaults(func=check_command)

    # Commit command
    commit_parser = subparsers.add_parser(
        "commit",
        help="Analyze registry changes and create a git commit with detailed summary",
    )
    commit_parser.set_defaults(func=commit_command)

    # Export-manifests command
    export_parser = subparsers.add_parser(
        "export-manifests",
        help="Convert Parquet registry files to Pooch-format text manifests for backwards compatibility",
    )
    export_parser.add_argument(
        "input_dir",
        help="Directory containing registry.parquet and landmasks.parquet files",
    )
    export_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for text manifest files (default: INPUT_DIR/registry)",
    )
    export_parser.set_defaults(func=export_manifests_command)

    # File-scan command
    file_scan_parser = subparsers.add_parser(
        "file-scan",
        help="Recursively scan year directories for embedding tiles and generate an inventory parquet file",
    )
    file_scan_parser.add_argument(
        "input_dir",
        help="Base directory containing year subdirectories (e.g., 2024/, 2023/) with embedding tiles",
    )
    file_scan_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet file path (default: INPUT_DIR/embedding_inventory.parquet)",
    )
    file_scan_parser.set_defaults(func=file_scan_command)

    # File-check command
    file_check_parser = subparsers.add_parser(
        "file-check",
        help="Check multiple inventory parquet files for duplicate year/lon/lat coordinates",
    )
    file_check_parser.add_argument(
        "parquet_files",
        nargs="+",
        help="Parquet files to check for duplicates (output from file-scan command)",
    )
    file_check_parser.set_defaults(func=file_check_command)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
