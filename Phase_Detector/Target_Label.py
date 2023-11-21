"""
This is a file to define the target label for the phase detector.

It is an interactive interface to label the phase.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets

def interactive_labeling(row_value, idx):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    ax.set_title("Select a phase for each point")

    ax.text(0.5, 0.7, row_value, ha='center', va='center', fontsize=20)

    phase_btns = plt.axes([0.1, 0.1, 0.6, 0.2])
    cancel_btns = plt.axes([0.75, 0.1, 0.1, 0.2])
    back_btns = plt.axes([0.88, 0.1, 0.1, 0.2])

    phase_choices = ["blanco", "AP", "PVP", "Other"]
    radio = widgets.RadioButtons(phase_btns, phase_choices, activecolor='blue')
    cancel_button = widgets.Button(cancel_btns, "Cancel", color='red')
    back_button = widgets.Button(back_btns, "Back", color='yellow')

    chosen_phase = [None]

    def on_radio_clicked(label):
        chosen_phase[0] = label
        plt.close()

    def on_cancel_clicked(event):
        chosen_phase[0] = "Cancel"
        plt.close()

    def on_back_clicked(event):
        chosen_phase[0] = "Back"
        plt.close()

    radio.on_clicked(on_radio_clicked)
    cancel_button.on_clicked(on_cancel_clicked)
    back_button.on_clicked(on_back_clicked)

    plt.tight_layout()
    plt.show()
    return chosen_phase[0]

csv_file = "Phase_Detector\CILM_xnatsort_phase_labels_20230323.csv"
df = pd.read_csv(csv_file)

# 使用函数
col_name = 'Series_description'
if 'Phase' not in df.columns:
    df['Phase'] = None

idx = 0
while idx < len(df):
    row = df.iloc[idx]
    if pd.notna(row['Phase']):
        idx += 1
        print(row['Phase'],'6666')
        continue
    else:
        print(3)
    choice = interactive_labeling(row[col_name], idx)
    
    if choice == "Cancel":
        break
    elif choice == "Back":
        idx -= 1 if idx > 0 else 0  # 如果已经是第一行，不做任何操作
        continue
    df.at[idx, 'Phase'] = choice
    idx += 1

print(df)

# 将更改写回原始CSV文件
df.to_csv(csv_file, index=False)
#print(type(df.iloc[100]['Phase']))

