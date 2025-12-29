import csv

def write_header(df, header_type):
    """
    Tworzy nagłówek pliku csv do zbierania danych
    :param df: output file
    :param header_type: calibration/prediction
    :return:
    """
    if header_type == 'calibration':
        with open(df, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['target_x', 'target_y', 'l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw', 'roll','eye_aspect_ratio'])
    elif header_type == 'prediction':
        with open(df, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw', 'roll', 'eye_aspect_ratio'])
    else:
        raise ValueError('Header type must be either calibration or prediction')


def save_calibration_data(df, point, l_relative_x, l_relative_y, r_relative_x,r_relative_y, pitch, yaw, roll,blink_ratio):
    """
    Zapisuje dane kalibracyjne do pliku csv
    :param df: output file
    """
    with open(df, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            point[0], point[1],
            l_relative_x, l_relative_y,
            r_relative_x, r_relative_y,
            pitch, yaw, roll,
            blink_ratio
        ])

def save_session_data(df, l_relative_x, l_relative_y, r_relative_x,r_relative_y, pitch, yaw, roll,blink_ratio):
    """
    Zapisuje dane sesji do pliku csv
    :param df: output file
    """
    with open(df, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            l_relative_x, l_relative_y,
            r_relative_x, r_relative_y,
            pitch, yaw, roll,
            blink_ratio
        ])