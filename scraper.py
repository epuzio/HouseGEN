import requests, sys, pandas as pd, os, tensorflow as tf
from bs4 import BeautifulSoup
from tensorflow.io import TFRecordWriter

'''
Web scraper to get images from Zillow. 
If 403 error encountered, use a virtual environment and/or switch the header.
'''

def scrape_zillow(locations):
    urls = []
    search = ["fsbo", "for_sale"] #for sale by owner, foreclosure, for sale
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


# def read_tensor_from_image_url(url,
#                                input_height=299,
#                                input_width=299,
#                                input_mean=0,
#                                input_std=255):
#     image_reader = tf.image.decode_jpeg(
#         requests.get(url).content, channels=3, name="jpeg_reader")
#     img_as_binary = tf.cast(image_reader, tf.float32)
#     dims_expander = tf.expand_dims(float_caster, 0)
#     resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
#     normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])

#     with tf.Session() as sess:
#         return sess.run(normalized)

def csv2tfr():
    df = pd.read_csv('zillow_urls.csv') #dataframe to store csv file
    os.makedirs('images', exist_ok=True) #make directory if one doesn't exist
    county_name = ""
    n_img = 0

    for index, row in df.iterrows(): #for each row in csv
        if row["URL"].startswith("#"): #change name
            county_name = row["URL"][5:]
            n_img = 0
        else:
            image_name = county_name + f'_{n_img}.jpg'
            image_path = f'images/{image_name}'

            response = requests.get(row["URL"])
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                n_img += 1
            else:
                print(f'Failed to download: {row["Image URL"]}')


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To scrape url by location: python3 scraper.py county-st county-st...")
        print("To convert csv to tfr: python3 scraper.py csv2tfr")
    else:
        if sys.argv[1] == "csv2tfr":
            csv2tfr()
        else:
            locations = sys.argv[1:]
            scrape_zillow(locations)