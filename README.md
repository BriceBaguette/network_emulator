# network_emulator

To run the web interface in docker: 

```
docker build -t netsim-dash-app .   
```
```
docker run -p 8050:8050 netsim-dash-app
```
or you can simply execute in your interpreter with:

```
python pip install -r requirements.txt
```
and

```
python ./src/app.py
```