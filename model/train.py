import dvc.api


def load_data():
    url = 'https://github.com/robert-min/bike_share_mlflow'
    branch = 'feature/eda-process'
    
    with dvc.api.open(
        path = 'data/train.csv',  ## 데이터 경로
        repo = url,  ## github repo 경로,
        rev =  branch ## 현재는 branch 기준
    ) as f:
        train = f.read()
    
    
    with dvc.api.open(
        path = 'data/train.csv',
        repo = url,
        rev = branch
    ) as f:
        test = f.read()
        
    if train and test:
        return train, test
    else:
        Exception("dvc load data error")

def main():
    train, test = load_data()
    print(test)

if __name__ == "__main__":
     main()