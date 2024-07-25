import cdsapi
from argparse import ArgumentParser

# Example usage: to download 64 months in 32 files for 5years4months
# python experiments/download_era5.py "2013-08" "2018-12" -m 2

c = cdsapi.Client()

def retrieve(year, months):
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                #'10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'skin_temperature', #'surface_pressure', 'total_precipitation',
            ],
            'year': year,
            'month': months,
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
        },
        f'era5_{year}_{months.join("-")}.nc'
        )
    
def get_year_month(start_year, start_month, increments):
    new_year = start_year + (start_month + increments) // 12
    new_month = (start_month + increments) % 12
    return new_year, new_month

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("date_range", type=str, nargs=2)
    parser.add_argument("-m", "--months_per_file", type=int, default=2)
    args = parser.parse_args()
    
    start_year, start_month = map(int, args.date_range[0].split("-"))
    end_year, end_month = map(int, args.date_range[1].split("-"))

    assert 12 % args.months_per_file == 0, "Year should be divible by months per file"

    num_chunks = ((end_year * 12 + end_month) - (start_year * 12 + start_month)) / args.months_per_file
    print(num_chunks)
    assert num_chunks.is_integer(), (
        "The date range must be a multiple of the number of months per file"
    )
    for chunk in range(int(num_chunks)):
        year, month = get_year_month(start_year, start_month, chunk * args.months_per_file)
        
        months = [str(m).zfill(2) for m in range(month, month + args.months_per_file)]
        print(str(year), months)
        retrieve(str(year), month)