import os
from datetime import datetime

def save_plot_with_timestamp(plt, title, folder_path):
    print("Saving plot...")
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f'{current_time}_{title}.png'
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path)

    return file_path