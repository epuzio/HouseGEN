import requests, sys, pandas as pd
from bs4 import BeautifulSoup

'''
Web scraper to get images from Zillow by location. It worked exactly once.
To do: resolve error 403 from Zillow.
'''

def scrape_zillow(locations):
    urls = []
    search = ["fsbo", "for_sale/fore_lt", "for_sale"] #for sale by owner, foreclosure, for sale
    for l in locations:
        for s in search:
            url = f"https://www.zillow.com/{l}/homes/{s}/*_p/$"
            headers = { #user agent for epuzio's browser
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            print("Status:", response.status_code)
            if response.status_code == 200: #web page fetched successfully
                soup = BeautifulSoup(response.content, "html.parser")
                img_tags = soup.find_all('img', src=lambda x: x and x.startswith('https://photos.zillowstatic.com/fp/'))
                # Extract src attribute from each img tag
                image_urls = [img['src'] for img in img_tags]
                urls.extend(image_urls)
            else:
                print("Failed to fetch page for location", l, "search criteria", s)
        df = pd.DataFrame(urls)
        df.to_csv(f"zillow_urls.csv", index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To scrape url by location: python3 scraper.py county-st county-st...")
    else:
        locations = sys.argv[1:]
        scrape_zillow(locations)
