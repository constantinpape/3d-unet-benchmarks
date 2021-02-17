import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--table', default='./results/embl.csv')
parser.add_argument('-s', '--split', default=1, type=int)
args = parser.parse_args()

df = pd.read_csv(args.table)

if bool(args.split):
    precisions = pd.unique(df['Precision'])
    for prec in precisions:
        df_prec = df[df['Precision'] == prec]
        print(f"**{prec.capitalize()}-precision**:")
        train_blocks = pd.unique(df_prec['Blocks_train'])
        inf_blocks = pd.unique(df_prec['Blocks_inf'])
        if len(train_blocks) > 1 or len(inf_blocks) > 1:
            print("Inconsistent block size!")
        print(f"Block shapes: {train_blocks[0]} (training), {inf_blocks[0]} (inference)")
        print()
        df_prec = df_prec.drop(axis=1,
                               labels=['Precision', 'Blocks_train', 'Blocks_inf'])
        md = df_prec.to_markdown(index=False)
        print(md)
        print()
else:
    df = df.drop(axis=1, labels=['Blocks_train', 'Blocks_inf'])
    md = df.to_markdown(index=False)
    print(md)
