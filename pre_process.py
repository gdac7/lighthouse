from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import IsolationForest
import pandas as pd

class PreProcess:
    def __init__(self, data, irrelevant_columns=None, encoder_type="ordinal_encoder", cat_columns=[], name_columns=None, remove_outliers=False):
        self.X, self.Y = self.sep_data(data)
        self.irr_col = irrelevant_columns
        self.cat_columns = cat_columns
        self.remove_outliers = remove_outliers
        self.name_columns = name_columns
        self.encoder_type = encoder_type

    

    def sep_data(self, data):
        X = data.drop("price", axis=1)
        Y = data['price'].copy()
        return X, Y

    def __set_encoder(self, typ):
        if typ == "one_hot_encoder":
            self.encoder = OneHotEncoder(handle_unknown="ignore")
        else:
            self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)



    def __set_pipeline(self):
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="constant", fill_value=0)),
            ('std_scaler', StandardScaler()),
        ])

        transformers = [
            ("num", num_pipeline, self.__num_attrs),
            ("cat", self.encoder, self.cat_columns)
        ]



        full_pipeline = ColumnTransformer(transformers)

        self.pipe = full_pipeline
            
    def __categorize_names(self):
        expensive_words = {0:"superbowl", 1: "film", 2: "event", 3: "photo", 4: "luxury"}
        def map_exp_words(name):
            name = str(name)
            for key, word in expensive_words.items():
                if word in name.lower():
                    return key
            return -1
        self.X['has_event_ref'] = self.X['nome'].apply(map_exp_words)
        self.X = self.X.drop(columns=self.name_columns)

                

    
    def __remove_outliers(self):
        isolation_forest = IsolationForest(random_state=42)
        outlier_pred = isolation_forest.fit_predict(pd.DataFrame(self.Y))
        self.X = self.X[outlier_pred == 1]
        self.Y = self.Y[outlier_pred == 1]

    
    def __sep_num_cat(self):
        self.__data_num = self.X.drop(columns=self.cat_columns)
        self.__num_attrs = list(self.__data_num)
        if self.name_columns:
            #Name columns não é categórico. Um processamento diferente deve ser feito
            vals_to_remove = set(self.name_columns)
            self.__num_attrs = [col for col in self.__num_attrs if col not in vals_to_remove]
        
        

    def __remove_irrelevant(self):
        self.X = self.X.drop(columns=self.irr_col)

    def __central_park_proximity(self):
            central_park_geo = {
                'latitude_min': 40.758112,
                'latitude_max': 40.807937,
                'longitude_min': -74,
                'longitude_max': -73.968656 
            }
            def prox_centralpark(row):
                cond1 = row['latitude'] >= central_park_geo['latitude_min'] and row['latitude'] <= central_park_geo['latitude_max']
                cond2 = row['longitude'] >= central_park_geo['longitude_min'] and row['longitude'] <= central_park_geo['longitude_max']
                if cond1 and cond2: 
                    return 1
                return 0

            self.X['near_central_park'] = self.X.apply(prox_centralpark, axis=1)
    

    def pre_process(self):
        self.__remove_irrelevant()
        self.__central_park_proximity()
        if self.remove_outliers:
            self.__remove_outliers()
        self.__sep_num_cat()
        self.__set_encoder(self.encoder_type)
        if self.name_columns:
            self.__categorize_names()
        self.__set_pipeline()
        self.X_prep = self.pipe.fit_transform(self.X)

        

