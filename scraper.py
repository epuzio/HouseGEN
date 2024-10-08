import requests, sys, pandas as pd, os, tensorflow as tf, time, csv
from bs4 import BeautifulSoup
from tensorflow.io import TFRecordWriter
from tensorflow.train import Example, Features, Feature, BytesList

'''
Web scraper to get images from Zillow. 
If 403 error encountered, use a virtual environment and/or switch the header.
'''

def scrape_zillow(locations):
    urls = []
    search = ["fsbo", "foreclosed", "for_sale"] #for sale by owner, foreclosure, for sale
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
            print("Response status code:", response.status_code) #if it's 200, success. if it's 403, forbidden (probably header issue)
            if response.status_code == 200: #web page fetched successfully
                soup = BeautifulSoup(response.content, "html.parser")
                img_tags = soup.find_all('img', src=lambda x: x and x.startswith('https://photos.zillowstatic.com/fp/'))
                image_urls = [img['src'] for img in img_tags if "_web_" not in img['src']] #get src from image tag, ignore non-house images
                urls.extend(image_urls)
            else:
                print("Failed to fetch page for location:", l, ", search criteria:", s)
        df = pd.DataFrame(urls)
        df.to_csv("zillow_urls.csv", mode="a", header=False, index=False)

def scrape_petfinder(locations): #copy+pasting because we aren't searching by multiple listing types
    urls = []
    for l in locations:
        with open("petfinder_urls.csv", "a") as file:
            file.write(f"# -- {l}\n")
        url = f"https://www.petfinder.com/search/cats-for-adoption/us/{l[-2:]}/{l[0:len(l)-3]}/"
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
        if response.status_code == 200: # web page fetched successfully
            soup = BeautifulSoup(response.content, "html.parser")
            #find parent div of the pfdc-animal-search-results element
            result_divs = soup.find('pfdc-animal-search-results').find_all('div', class_='result')

            print(result_divs)

            img_tags = soup.find_all('img')
            print("Number of images found:", len(img_tags))
            print(img_tags)
            # for img_tag in img_tags:
            #     srcset = img_tag.get('srcset')
            #     if srcset:
            #         print(srcset)
            image_urls = [img['srcset'] for img in img_tags] #get src from image tag, ignore non-house images
            urls.extend(image_urls)
        else:
            print("Failed to fetch page for location:", l)
        df = pd.DataFrame(urls)
        df.to_csv("petfinder_urls.csv", mode="a", header=False, index=False)

def image_to_jpg(image_data):
    """Convert image data to JPEG format."""
    image = Image.open(io.BytesIO(image_data))
    with io.BytesIO() as output:
        image.save(output, format="JPEG")
        jpeg_data = output.getvalue()
    return jpeg_data

def csv2tfr(tfrecord=True):
    df = pd.read_csv('zillow_urls.csv') #dataframe to store csv file
    os.makedirs('images', exist_ok=True) #make directory if one doesn't exist
    os.makedirs('tfr', exist_ok=True) #make directory if one doesn't exist
    county_name = ""
    n_img = 0
    n_counties = 0
    lines = 0
    # crop_bar = tf.zeros([128, 512, 3], dtype=tf.uint8) #bar to add to image so that it's 512x512
    crop_bar = tf.zeros([150, 596, 3], dtype=tf.uint8) #bar to add to image so that it's 512x512

    with open("zillow_urls.csv", 'r') as file: #print number of URLs
        for f in file:
            if f[0] == "#":
                lines += 1
    print("Reading images from", lines, "counties listed in csv ...")
    start_time = time.time()

    with TFRecordWriter("houses.tfrecord") as writer:
        for index, row in df.iterrows(): #for each row in csv
            if row["URL"].startswith("#"): #change name
                county_name = row["URL"][5:]
                n_img = 0
                n_counties += 1
                print("Saving images from county", n_counties, "...", end="\r")
            else:
                image_name = county_name + f'_{n_img}.jpg'
                image_path = f'images/{image_name}'

                response = requests.get(row["URL"])
                if response.status_code == 200:
                    if not tfrecord: #save as jpg
                        with open(image_path, 'wb') as f:
                            f.write(response.content)
                    else: #save as tfrecord
                        image_as_tensor = tf.image.decode_image(response.content, channels=3)
                        img_with_bar = tf.image.resize(tf.concat([image_as_tensor, crop_bar], axis=0), [512, 512])
                        img_with_bar = tf.cast(img_with_bar, tf.uint8)
                        feature = {
                            'image': Feature(bytes_list=BytesList(value=[img_with_bar.numpy().tobytes()]))
                        }
                        example = Example(features=Features(feature=feature))
                        tfr_example = example.SerializeToString()
                        writer.write(tfr_example)
                    n_img += 1
                else:
                    print(f'Failed to download: {row["Image URL"]}')
    print("Saved houses from", n_counties, "counties, completed in", time.time() - start_time, "seconds.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To scrape url by location: python3 scraper.py county-st county-st...")
        print("To convert csv to tfr: python3 scraper.py csv2tfr")
    else:
        if sys.argv[1] == "csv2tfr":
            csv2tfr()
        if sys.argv[1] == "petfinder":
            locations = sys.argv[2:]
            scrape_petfinder(locations)
        else:
            locations = sys.argv[1:]
            scrape_zillow(locations)
