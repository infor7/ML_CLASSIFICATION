# ML_CLASSIFICATION

## How to start development?
```
git clone https://github.com/infor7/ML_CLUSTERING.git
cd ML_CLUSTERING
virtualenv venv --python=python3.6
source venv/bin/activate
pip install -r requirements.txt
```

Add a new directory under app/scripts

Place your code inside it. Code must have function named execute.

Put into it base cotent
```
def execute(ax=None, **kwargs):
    ax = ax or plt.gca()
    line, = ax.plot (np.arange(0.0, 5.0, 0.02))
```

Add record to xml undex app/config/config.xml

Run 
```
python app/start.py
```
