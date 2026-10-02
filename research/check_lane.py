import pandas as pd
import numpy as np

def check_lex_fw():
    df_train = pd.read_csv("train-test.csv")
    df_val = pd.read_csv("validation.csv")
    
    train_lex = df_train[(df_train['pickup'] == 'Lexington') & (df_train['delivery'] == 'Fort Wayne')]
    val_lex = df_val[(df_val['pickup'] == 'Lexington') & (df_val['delivery'] == 'Fort Wayne')]
    
    print("=== TRAIN LEXINGTON -> FORT WAYNE ===")
    print(train_lex[['date', 'equipment', 'distance', 'weight', 'quote_signal', 'market_index', 'posted_rate']])
    
    print("\n=== VAL LEXINGTON -> FORT WAYNE ===")
    print(val_lex[['load_id', 'date', 'equipment', 'distance', 'weight', 'quote_signal', 'market_index']])

    # What is the average quote_signal for Dry Van on Lexington -> Fort Wayne?
    train_dv = train_lex[train_lex['equipment'] == 'Dry Van']
    val_dv = val_lex[val_lex['equipment'] == 'Dry Van']
    print("\nTrain Dry Van quote_signals:", train_dv['quote_signal'].values)
    print("Val Dry Van quote_signals:", val_dv['quote_signal'].values)
    
    # Is quote_signal predictable from (distance, equipment, weight, lane)?
    # Let's train a model to predict quote_signal or check its regression formula!
    print("\nOverall Dry Van quote_signal mean:", df_train[df_train['equipment'] == 'Dry Van']['quote_signal'].mean())
    print("Dry Van Lexington -> Fort Wayne quote_signal mean:", 
          pd.concat([train_dv, val_dv])['quote_signal'].mean())

if __name__ == "__main__":
    check_lex_fw()
