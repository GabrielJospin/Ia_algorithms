# AI Algorithms

## To run <br>

To run, you will need firstly config the ```config.json```. The attributes of the file is: <br>
- ```path```: the path of the file .csv where is the database
- ```separator```: the type of separator used in .csv
- ```numOut```: number of outs in expected
- ```classifier```: The Classifier you want to use
<br>

after this run this on terminal:

```
python main.py
```

## The Result

You can read the result of the run in ```./log/${classifier}/${datetime}``` 

## Classifiers Avaible

- Decision Tree as ```DecisionTree```
- Linear Regression as ```LinearRegression```

## To Future

- New Classifiers
- Files to explain all algorithms
