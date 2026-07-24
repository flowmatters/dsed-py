#!/usr/bin/env python
'''
Stamp the correct CRS onto the network GeoJSON files written alongside converted
OpenWater models.

Models converted before the CRS fix have .links.json / .nodes.json / .catchments.json
files that carry no CRS at all, so every reader falls back to the GeoJSON default of
EPSG:4326 -- while the coordinates are actually in the Source project's projection
(EPSG:3577, GDA94 / Australian Albers, for the GBR models).

This rewrites those files with the CRS declared. Coordinates are not moved, so
anything already working against the projected coordinates keeps working.

Usage:

    python set_network_crs.py --crs EPSG:3577 migration/BU.h5 migration/WT.h5
    python set_network_crs.py --crs EPSG:3577 migration/          # all models in a dir
    python set_network_crs.py --crs EPSG:3577 --dry-run migration/
'''
import argparse
import os
import sys

import geopandas as gpd

SUFFIXES = ['.links.json', '.nodes.json', '.catchments.json']


def model_bases(paths):
    '''
    Expand the command line arguments into model base names (path minus the .h5).
    '''
    bases = []
    for path in paths:
        if os.path.isdir(path):
            bases += [os.path.join(path, f[:-3]) for f in sorted(os.listdir(path)) if f.endswith('.h5')]
        elif path.endswith('.h5'):
            bases.append(path[:-3])
        else:
            bases.append(path)
    return bases


def set_crs(fn, crs, force=False, dry_run=False):
    '''
    Rewrite a single GeoJSON file with crs declared. Returns a status string.
    '''
    gdf = gpd.read_file(fn)

    # geopandas reports EPSG:4326 for a GeoJSON with no crs member, so we can't tell
    # "genuinely 4326" from "unspecified" by reading the frame. Check the file itself.
    import json
    with open(fn) as fp:
        declared = json.load(fp).get('crs')

    if declared is not None and not force:
        return 'skipped (already declares %s)' % declared['properties']['name']

    if dry_run:
        return 'would set %s (%d features)' % (crs, len(gdf))

    gdf = gdf.set_crs(crs, allow_override=True)
    gdf.to_file(fn, driver='GeoJSON')
    return 'set %s (%d features)' % (crs, len(gdf))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('paths', nargs='+', help='Model .h5 files, model base names, or directories containing them')
    parser.add_argument('--crs', required=True, help='CRS to declare, eg EPSG:3577')
    parser.add_argument('--force', action='store_true', help='Overwrite a CRS that is already declared')
    parser.add_argument('--dry-run', action='store_true', help='Report what would change without writing')
    args = parser.parse_args()

    bases = model_bases(args.paths)
    if not bases:
        print('No models found', file=sys.stderr)
        return 1

    for base in bases:
        print(os.path.basename(base))
        for suffix in SUFFIXES:
            fn = base + suffix
            if not os.path.exists(fn):
                print('  %-18s missing' % suffix)
                continue
            print('  %-18s %s' % (suffix, set_crs(fn, args.crs, args.force, args.dry_run)))

    return 0


if __name__ == '__main__':
    sys.exit(main())
