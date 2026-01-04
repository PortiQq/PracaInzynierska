import csv

def write_header(df, header_type):
    """
    Tworzy nagłówek pliku csv do zbierania danych
    :param df: output file
    :param header_type: calibration/validation/session
    :return:
    """
    if header_type == 'calibration':
        with open(df, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['target_x', 'target_y', 'l_rel_x', 'l_rel_y', 'r_rel_x', 'r_rel_y', 'pitch', 'yaw', 'roll','eye_aspect_ratio'])
    elif header_type == 'validation':
        with open(df, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['point_x', 'point_y', 'screen_x', 'screen_y'])
    elif header_type == 'session':
        with open(df, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'screen_x', 'screen_y'])
    else:
        raise ValueError('Header type must be either calibration/validation/session')


def save_calibration_data(df, point, l_relative_x, l_relative_y, r_relative_x,r_relative_y, pitch, yaw, roll,blink_ratio):
    """
    Zapisuje dane kalibracyjne do pliku csv
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


def save_validation_data(df, point_x, point_y, screen_x, screen_y):
    """
    Zapisuje dane sesji do pliku csv
    """
    with open(df, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            point_x, point_y,
            screen_x, screen_y
        ])


def save_session_data(df, timestamp, screen_x, screen_y):
    """
    Zapisuje dane sesji do pliku csv
    """
    with open(df, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, screen_x, screen_y
        ])