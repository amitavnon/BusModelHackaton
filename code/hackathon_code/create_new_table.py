from .preprocess_data import *


def create_trip_table_test(df: pd.DataFrame):
    trip_df = df.groupby('trip_id_unique').agg({
        'trip_id': 'first',
        'line_id': 'first',
        'direction': 'first',
        'alternative': 'first',
        'cluster': 'first',
        'latitude': 'mean',
        'longitude': 'mean',
        'passengers_up': 'sum',
        'trip_id_unique_station': 'count',
        'passengers_continue': 'max',
        'passengers_continue_menupach': 'mean',
        'arrival_total_seconds': ['max', 'min'],
    }).reset_index()

    trip_df['start_time'] = trip_df['arrival_total_seconds']['min']

    # Flatten the MultiIndex columns
    trip_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                       for col in trip_df.columns]

    # Rename columns to remove '_first' suffix for single aggregation columns
    trip_df.rename(columns=lambda x: x.replace('_first', ''), inplace=True)
    return trip_df


def create_trip_table(df: pd.DataFrame):
    trip_df = df.groupby('trip_id_unique').agg({
        'trip_id': 'first',
        'line_id': 'first',
        'direction': 'first',
        'alternative': 'first',
        'cluster': 'first',
        'latitude': 'mean',
        'longitude': 'mean',
        'passengers_up': 'sum',
        'trip_id_unique_station': 'count',
        'passengers_continue': 'max',
        'passengers_continue_menupach': 'mean',
        'arrival_total_seconds': ['max', 'min'],
    }).reset_index()

    trip_df['start_time'] = trip_df['arrival_total_seconds']['min']

    # Calculate duration
    trip_df['duration'] = trip_df['arrival_total_seconds']['max'] - \
                          trip_df['arrival_total_seconds']['min']
    trip_df.drop(['arrival_total_seconds'], axis=1, inplace=True)
    # Flatten the MultiIndex columns
    trip_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                       for col in trip_df.columns]

    # Rename columns to remove '_first' suffix for single aggregation columns
    trip_df.rename(columns=lambda x: x.replace('_first', ''), inplace=True)

    trip_df = trip_df[trip_df['duration_'] > 0]
    trip_df['duration_'] = trip_df['duration_'] / 60
    trip_df = trip_df[trip_df['duration_'] < 1000]


    return trip_df.drop(['duration_', 'trip_id_unique_'],
                        axis=1), trip_df.duration_


def create_trip_table_test_data(df: pd.DataFrame):
    trip_df = df.groupby('trip_id_unique').agg({
        'trip_id': 'first',
        'line_id': 'first',
        'direction': 'first',
        'alternative': 'first',
        'cluster': 'first',
        'latitude': 'mean',
        'longitude': 'mean',
        'passengers_up': 'sum',
        'trip_id_unique_station': 'count',
        'passengers_continue': 'max',
        'passengers_continue_menupach': 'mean',
        'arrival_total_seconds': ['max', 'min'],
    }).reset_index()
    trip_df['start_time'] = trip_df['arrival_total_seconds']['min']
    trip_df.drop(['arrival_total_seconds'], axis=1, inplace=True)
    # Flatten the MultiIndex columns
    trip_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                       for col in trip_df.columns]

    # Rename columns to remove '_first' suffix for single aggregation columns
    trip_df.rename(columns=lambda x: x.replace('_first', ''), inplace=True)

    return trip_df
