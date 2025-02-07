import joblib
import pandas as pd
from pre_process import PreProcess
def load_df(path):
    return pd.read_csv(path)



if __name__ == "__main__":
    model = joblib.load("linear_regression_model.pkl")
    df = load_df("./data/teste_indicium_precificacao.csv")
    some_data = df[:10]
    some_labels = df[:10]['price'].copy()
    some_data_prep =  PreProcess(
        data=some_data,
        irrelevant_columns=['nome', 'host_id', 'host_name', 'id', 'ultima_review'],
        encoder_type='ordinal_encoder',
        cat_columns=['bairro', 'bairro_group', 'room_type']
    )
    some_data_prep.pre_process()
    predictions = model.predict(some_data_prep.X_prep)
    predictions = [int(x) for x in predictions]
    print("Predições: ",list(predictions))
    print("Valores reais: ",list(some_labels))
