# walnutpi-debs
此项目保存了核桃派官方apt源内的所有包
## clone
```
git clone https://github.com/walnutpi/walnutpi-debs.git
```


## 1. 生成
生成一个`server`文件夹，输出相关文件
```
./gen.sh
```
## 2. 配置nginx
安装nginx
```
sudo apt install nginx
```

打开配置文件
```bash
sudo  vim /etc/nginx/sites-enabled/default
```
在`server`中添加一个`location`项，将路径指向本项目的outpu文件夹
```
    location /debian {
        alias /xxx/walnutpi-debs/server;
        autoindex on;
    }
```

重启nginx服务
```bash
sudo systemctl restart nginx.service
```

## 3. 配置apt源
打开用于设置apt源的配置文件
```
sudo vim /etc/apt/sources.list
```
插入源，把其中的xxxx改成自己的ip地址
```bash
deb [trusted=yes] http://xxxxx/debian/ bookworm main
```