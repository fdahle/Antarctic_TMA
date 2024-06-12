# Library imports
import base64
import json
import os
import netrc
from urllib.parse import urlparse, urlencode
from urllib.request import urlopen, Request, build_opener, HTTPCookieProcessor

# Local imports
import src.base.load_credentials as lc

# Constants
CMR_URL = 'https://cmr.earthdata.nasa.gov'
URS_URL = 'https://urs.earthdata.nasa.gov'
CMR_PAGE_SIZE = 2000
CMR_FILE_URL = f'{CMR_URL}/search/granules.json?provider=NSIDC_ECS&' \
               f'sort_key[]=start_date&sort_key[]=producer_granule_id&' \
               f'scroll=true&page_size={CMR_PAGE_SIZE}'


def load_ice_bridge(bbox, download_fld, short_name='IODMS1B', version='1',
                    time_start='2009-10-16T00:00:00Z',
                    time_end='2018-04-19T23:59:59Z'):
    # Authentication and URL setup
    def get_credentials():
        try:
            info = netrc.netrc()
            username, account, password = info.authenticators(urlparse(URS_URL).hostname)
            return base64.b64encode(f'{username}:{password}'.encode('ascii')).decode('ascii')
        except (Exception,):
            # username = input('Earthdata username: ')
            # password = getpass('Earthdata password: ')
            username, password = lc.load_credentials('earthdata')
            return base64.b64encode(f'{username}:{password}'.encode('ascii')).decode('ascii')

    def build_query_url():
        query = {
            'short_name': short_name,
            'version': version.zfill(3),
            'temporal': f'{time_start},{time_end}',
            'bounding_box': bbox
        }
        return f"{CMR_FILE_URL}&{urlencode(query)}"

    def download_files(_urls):
        credentials = get_credentials()
        if not os.path.exists(download_fld):
            os.makedirs(download_fld)
        for url in _urls:
            filename = os.path.join(download_fld, url.split('/')[-1])
            print(f'Downloading {filename}...')
            _req = Request(url)
            _req.add_header('Authorization', f'Basic {credentials}')
            try:
                with urlopen(_req) as _response, open(filename, 'wb') as out_file:
                    out_file.write(_response.read())
            except (Exception,) as e:
                print(f'Failed to download {filename}: {e}')

    # Fetch data
    query_url = build_query_url()

    print(query_url)

    req = Request(query_url)
    with urlopen(req) as response:
        search_page = json.loads(response.read().decode('utf-8'))

    # Extract download URLs
    urls = [entry['links'][0]['href'] for entry in search_page['feed']['entry'] if 'links' in entry and entry['links']]

    print(urls)

    download_files(urls)

# Example usage:
# load_ice_bridge('-69.48,-69.31,-61.02,-64.61')
