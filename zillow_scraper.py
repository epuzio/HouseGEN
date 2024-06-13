import requests, sys, pandas as pd, os, tensorflow as tf, time, numpy as np, zipfile
from bs4 import BeautifulSoup
from PIL import Image
import io
from io import BytesIO
from tensorflow.io import TFRecordWriter
from tensorflow.train import Example, Features, Feature, BytesList
import imageio

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

def zip_files():
    """
    Zips all images in output_images to a .zip file
    """
    with zipfile.ZipFile("output_training_set_png.zip", "w") as zipf:
        for img in os.listdir("images"):
            if img.endswith(".jpg"):
                # Open the image
                image_path = os.path.join("images", img)
                image = Image.open(image_path)

                # Convert to PNG and save to BytesIO object
                png_data = BytesIO()
                image.save(png_data, format="PNG")
                png_data.seek(0)

                # Add the PNG data to the ZIP file
                png_filename = os.path.splitext(img)[0] + ".png"
                zipf.writestr(png_filename, png_data.read())
                

def csv2img(tfrecord=False):
    df = pd.read_csv('zillow_urls.csv') #dataframe to store csv file
    os.makedirs('images', exist_ok=True) #make directory if one doesn't exist
    os.makedirs('tfr', exist_ok=True) #make directory if one doesn't exist
    county_name = ""
    n_img = 0
    n_counties = 0
    lines = 0
    with open("Bar.jpg", "rb") as f: #unfathomably naive
        bar_image_data = f.read()
    bar_image = tf.io.decode_jpeg(bar_image_data, channels=3)
    print("Bar image shape:", bar_image.shape)

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
                    with open(image_path, 'wb') as f:
                        image = Image.open(io.BytesIO(response.content))
                        new_size = (512, 383)
                        resized_image = image.resize(new_size, Image.LANCZOS)      
                        new_image_height = 512
                        new_image = Image.new("RGB", (512, new_image_height), "black")
                        new_image.paste(resized_image, (0, 192))
                        new_image.save(image_path)
                    n_img += 1
                else:
                    print(f'Failed to download: {row["Image URL"]}')
    print("Saved houses from", n_counties, "counties, completed in", time.time() - start_time, "seconds.")

# def augment_image(image, n_new_imgs=3):
#     '''
#     Apply image augmentations.
#     '''
#     new_imgs = [] 
#     image = tf.image.decode_jpeg(image, channels=3)
#     img1 = tf.image.random_flip_left_right(image)
#     img2 = tf.image.random_brightness(image, max_delta=0.2)
#     img2 = tf.image.random_contrast(img2, max_delta=0.2)
#     rotated = tf.image.rot90(image)
#     image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
#     image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
#     image = tf.image.random_hue(image, max_delta=0.2)
#     image = tf.image.encode_jpeg(tf.cast(image, tf.uint8))
#     return image

def zip2tfr():
    '''
    Save all images in output_images to a .tfrecord file - this is an efficient way 
    to store a dataset for the GAN. Tensorflow is imported here as it takes a long time to import
    and should only be imported when necessary.
    '''

    tfrecord_file_path = 'output.tfrecord'
    
    if not (os.path.exists("output_training_set.zip")):
        zip_files()
    count = 0
    with TFRecordWriter(tfrecord_file_path) as writer:
        with zipfile.ZipFile("output_training_set.zip", "r") as zip_ref:
            for file_name in zip_ref.namelist():
                image_data = zip_ref.read(file_name)
                
                feature = {
                    'image': Feature(bytes_list=BytesList(value=[image_data]))
                }
                
                example = Example(features=Features(feature=feature))
                writer.write(example.SerializeToString()) #SerializeToString turns the example into a binary string
                count += 1
    print(f"TFRecord file created from zip: {tfrecord_file_path}, {count} images written.")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("To scrape url by location: python3 zillow_scraper.py county-st county-st...")
        print("To convert csv to img: python3 zillow_scraper.py csv2tfr")
        print("To zip: python3 zillow_scraper.py zip")
        print("To convert zip to tfr: python3 zillow_scraper.py tfr")
    else:
        if sys.argv[1] == "csv2tfr":
            csv2img()
        elif sys.argv[1] == "zip":
            zip_files()
        elif sys.argv[1] == "tfr":
            zip2tfr()
        else:
            locations = sys.argv[1:]
            scrape_zillow(locations)
