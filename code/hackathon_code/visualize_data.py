from argparse import ArgumentParser
import matplotlib.pyplot as plt
from create_new_table import *


def pearson_correlation(feature, label, plot_title):
    corr = feature.cov(label) / (feature.std() * label.std())

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(feature, label, alpha=0.5)
    plt.title(f'{plot_title}\nPearson Correlation: {corr:.2f}')
    plt.xlabel('feature')
    plt.ylabel('Response')

    # Save plot to the specified output path
    plt.savefig(f'{plot_title}.png')
    plt.close()


def load_set(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path, encoding="ISO-8859-8")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    df = load_set(args.training_set)
    # Passengers up

    # Pearson correlations:
    df_processed = basic_preprocess(df)
    pearson_correlation(df_processed['station_id'],
                        df_processed['passengers_up'], "station to passenger")
    pearson_correlation(df_processed['total_door_open_time'],
                        df_processed['passengers_up'],
                        "door open time to passenger")
    pearson_correlation(df_processed['passengers_continue'],
                        df_processed['passengers_up'],
                        "passenger continue to passenger up")
    pearson_correlation(df_processed['passengers_continue_menupach'],
                        df_processed['passengers_up'],
                        "passengers continue menupach to passenger up")
    pearson_correlation(df_processed['mekadem_nipuach_luz'],
                        df_processed['passengers_up'],
                        "mekadem_nipuach_luz to passenger up")
    pearson_correlation(df_processed['longitude'],
                        df_processed['passengers_up'],
                        "longitude to passenger up")
    pearson_correlation(df_processed['latitude'],
                        df_processed['passengers_up'],
                        "latitude to passenger up")

    # Scatter plot passengers up vs door open time
    plt.figure(figsize=(8, 6))
    plt.scatter(df_processed['total_door_open_time'],
                df_processed['passengers_up'], color='blue', alpha=0.5)
    plt.title('passengers up vs door open time')
    plt.xlabel('door opening time (seconds)')
    plt.ylabel('number of passengers that got on')
    # Show the plot
    plt.savefig('passengers up vs door open time.png')
    plt.close()

    # Duration
    trip_data_X, duration = create_trip_table(df_processed)
    pearson_correlation(trip_data_X['start_time_'],
                        duration,
                        "start time of bus vs duration")
    pearson_correlation(trip_data_X['total_door_open_time_mean'],
                        duration,
                        "door open time vs duration")
    pearson_correlation(trip_data_X['passengers_up_sum'],
                        duration,
                        "passengers up vs duration")
    pearson_correlation(trip_data_X['trip_id_unique_station_count'],
                        duration,
                        "num stations vs duration")
    pearson_correlation(trip_data_X['mekadem_nipuach_luz'],
                        duration,
                        "mekadem_nipuach_luz vs duration")
