import os 
import urllib.request

def show_progress(block_num,block_size,total_size):
    downloaded=block_num*block_size
    finish_rate=downloaded/total_size
    if finish_rate>1:
        finish_rate=1

    progress='%.2f'%(finish_rate*100)
    bar=int(finish_rate*30)
    print('\r[{}] {}%'.format('#'*bar+'.'*(30-bar),progress))

def get_file(url,file_name=None):
    if file_name is None:
        file_name=url[url.rfind('/') + 1:]

    cache_dir=os.path.join(os.path.expanduser('~'),'.another-pytorch')
    file_path=os.path.join(cache_dir,file_name)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    if os.path.exists(file_path):
        return file_path
    
    print("Downloading: "+file_name)
    try:
        urllib.request.urlretrieve(url,file_path,show_progress)
    except (Exception,KeyboardInterrupt):
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print(" Done")
    return file_path

def pair(param):
    if isinstance(param,tuple) or isinstance(param,list):
        return param
    return (param,param)