import requests, sys, pandas as pd
from bs4 import BeautifulSoup

'''
Web scraper to get images from Zillow by location. It worked exactly once.
To do: resolve error 403 from Zillow.
In my defense: robots.txt states that the url /homes/fsbo/*_p/$ is allowed!
'''

def scrape_zillow(locations):
    urls = []
    search = ["fsbo", "for_sale/fore_lt", "for_sale"] #for sale by owner, foreclosure, for sale
    search = ["for_sale"] #for sale by owner, foreclosure, for sale
    for l in locations:
        for s in search:
            url = f"https://www.zillow.com/{l}/homes/{s}/$"
            print("URL:", url)
            headers = { #from https://scrapeops.io/web-scraping-playbook/403-forbidden-error-web-scraping/
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:98.0) Gecko/20100101 Firefox/98.0",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
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

    #cities: Lake-Havasu-City AZ, dolan springs az
