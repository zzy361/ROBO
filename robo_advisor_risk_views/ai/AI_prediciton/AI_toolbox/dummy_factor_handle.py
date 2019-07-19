import pandas as pd

def dummy_factor_handle(df,dummy_fields):
    for each in dummy_fields:

        dummies = pd.get_dummies(df.loc[:, each], prefix=each)
        df = pd.concat([df, dummies], axis=1)
    df.drop(dummy_fields, axis=1,inplace=True)
    return df
if __name__=="__main__":
    df = pd.read_excel("dummy_test.xlsx")
    dummy_fields=['male','color']
    df = dummy_factor_handle(df,dummy_fields)
    df.to_excel("df.xlsx")
