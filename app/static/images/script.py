import gdown
import tarfile
import gzip
import shutil

url = 'https://drive.google.com/file/d/1mQPzS2je9zXYkQpUm_OWM2YskjATjuzN/view?usp=share_link'
output = 'collected_images.zip'
gdown.download(url, output, quiet=False)


# open file
#file = tarfile.open('collected_images.tar.gz')
with gzip.open('collected_images.gz') as g:
    with open('images.zip', 'wb') as f_out:
        shutil.copyfileobj(g, f_out)

tarfile.open('images.zip', 'r:zip')
# extracting a specific file
"""file.extractall('.')
  
file.close()

import gzip
import shutil

with gzip.open('output.gz') as g:
    with open('output2.gz', 'wb') as f_out:
        shutil.copyfileobj(g, f_out)

tarfile.open('output2.gz', 'r:gz')"""