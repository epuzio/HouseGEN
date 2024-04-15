import requests, sys, pandas as pd, os
from bs4 import BeautifulSoup

'''
Web scraper to get images from Zillow. 
If 403 error encountered, use a virtual environment and/or switch the header.
'''

def scrape_zillow(locations):
    urls = []
    search = ["fsbo", "for_sale"] #for sale by owner, foreclosure, for sale
    # search = ["for_sale"] #for sale by owner, foreclosure, for sale
    for l in locations:
        with open("zillow_urls.csv", "a") as file:
            file.write(f"# -- {l}\n")
        for s in search:
            url = f"https://www.zillow.com/{l}/homes/{s}/*_p/$"
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
            print("Response status code:", response.status_code) #if it's 200, success. if it's 403.... :(
            if response.status_code == 200: #web page fetched successfully
                soup = BeautifulSoup(response.content, "html.parser")
                img_tags = soup.find_all('img', src=lambda x: x and x.startswith('https://photos.zillowstatic.com/fp/'))
                image_urls = [img['src'] for img in img_tags if "_web_" not in img['src']] #get src from image tag, ignore non-house images
                urls.extend(image_urls)
            else:
                print("Failed to fetch page for location:", l, ", search criteria:", s)
        df = pd.DataFrame(urls)
        df.to_csv("zillow_urls.csv", mode="a", header=False, index=False)

def csv2tfr():
    # Load the CSV file into a DataFrame
    df = pd.read_csv('your_csv_file.csv')

    # Create a directory to store the images
    os.makedirs('images', exist_ok=True)

    # Download each image from the URLs in the DataFrame
    for index, row in df.iterrows():
        url = row['Image URL']
        image_name = url.split('/')[-1]
        image_path = f'images/{image_name}'

        # Download the image
        response = requests.get(url)
        if response.status_code == 200:
            with open(image_path, 'wb') as f:
                f.write(response.content)
            print(f'Downloaded: {image_path}')
        else:
            print(f'Failed to download: {url}')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To scrape url by location: python3 scraper.py county-st county-st...")
    else:
        locations = sys.argv[1:]
        scrape_zillow(locations)

    #cities: Lake-Havasu-City AZ, dolan springs az
