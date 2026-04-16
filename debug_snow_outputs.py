from pathlib import Path
import pandas as pd
import rioxarray

root = Path.cwd()
output_summary_dir = root / 'outputs' / 'snow_summary'
output_cog_dir = root / 'outputs' / 'snow_cogs'
print('summary dir', output_summary_dir.exists())
print('cog dir', output_cog_dir.exists())
print('summary files', sorted([p.name for p in output_summary_dir.glob('*')]))

summary_csv = output_summary_dir / 'snow_daily_stats.csv'
print('summary csv exists', summary_csv.exists())
if summary_csv.exists():
    df = pd.read_csv(summary_csv)
    print(df.head())
    print('rows', len(df))
    print('snowy_pixels min/max', df['snowy_pixels'].min(), df['snowy_pixels'].max())
    print('snowy_pixels unique count', df['snowy_pixels'].nunique())

cog_paths = sorted(output_cog_dir.glob('snow_*.tif'))
print('COG count', len(cog_paths))
for p in cog_paths[:3]:
    ds = rioxarray.open_rasterio(str(p), masked=True).squeeze('band', drop=True)
    print('path', p.name)
    print('shape', ds.shape, 'dtype', ds.dtype)
    print('min/max', float(ds.min()), float(ds.max()))
    print('count == 200', int((ds == 200).sum().item()))
    print('count >= 0', int((ds >= 0).sum().item()))
    print('count masked', int(ds.count().item()))
    break
for p in [output_summary_dir / 'snow_days.tif', output_summary_dir / 'snow_frequency.tif']:
    ds = rioxarray.open_rasterio(str(p), masked=True).squeeze('band', drop=True)
    print('summary raster', p.name, 'shape', ds.shape, 'dtype', ds.dtype, 'min/max', float(ds.min()), float(ds.max()))
