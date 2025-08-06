# Library imports
import copy
import pandas as pd
import psycopg2
from typing import Optional, Union
from collections import Counter
from tqdm import tqdm

# Local imports
import src.base.connect_to_database as ctd

overwrite = True

def estimate_focal_length_text():
    """
    Estimates the focal length for a given image based on images with similar properties,
    or reconstructs it from OCR patterns if needed.

    Args:
        image_id (str): The ID of the image.
        use_estimated (bool): Whether to include estimated focal lengths in the stats.
        return_data (bool): Whether to return the supporting focal length DataFrame.
        focal_length_data (Optional[pd.DataFrame]): Preloaded focal length values.
        conn (Optional[Connection]): Optional database connection.

    Returns:
        float or (float, DataFrame) or None: Depending on flags and data availability.
    """

    conn = ctd.establish_connection()

    # get all images and focal lengths from the database
    sql_string = "SELECT image_id, focal_length, focal_length_estimated FROM images_extracted"
    data = ctd.execute_sql(sql_string, conn)

    # shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # remove all entries that have a focal length
    if overwrite is False:
        data = data[data['focal_length'].isnull()]

    for idx, row in (pbar := tqdm(data.iterrows(), total=data.shape[0])):

        # get the image id
        image_id = row['image_id']
        old_focal = row['focal_length']

        sql = (f"SELECT tma_number, view_direction, cam_id FROM images "
               f"WHERE image_id = '{image_id}'")
        img_data = ctd.execute_sql(sql, conn)

        tma = img_data["tma_number"].iloc[0]
        view = img_data["view_direction"].iloc[0]
        cam_id = img_data["cam_id"].iloc[0]

        if pd.isna(tma) or pd.isna(view) or pd.isna(cam_id):
            continue

        # Get focal_length_text for all images with the same flight and camera
        sql = f"""
            SELECT focal_length_text FROM images_extracted
            WHERE image_id IN (
                SELECT image_id FROM images
                WHERE tma_number = {tma}
                AND view_direction = '{view}'
                AND cam_id = {cam_id}
            )
            AND focal_length_text IS NOT NULL
        """
        all_texts = ctd.execute_sql(sql, conn)

        if all_texts.empty:
            continue

        text_list = all_texts['focal_length_text'].tolist()
        focal = _estimate_from_text_char_votes(text_list)

        if focal is not None and focal < 100:
            focal = focal + 100

        print(old_focal, focal)

        continue


def _estimate_from_text_char_votes(focal_length_texts: list[str]) -> Optional[float]:
    """
    Estimate focal length by majority vote at each position across OCR xxx.xxx strings.

    Args:
        focal_length_texts (list[str]): Semicolon-separated string lists.

    Returns:
        Optional[float]: Cleaned and reconstructed value.
    """
    all_candidates = []
    for entry in focal_length_texts:
        all_candidates.extend([
            s.strip() for s in entry.split(';')
            if len(s.strip()) == 7 and s.strip()[3] == '.'
        ])

    if not all_candidates:
        return None

    positions = list(zip(*all_candidates))
    most_common_chars = [Counter(pos).most_common(1)[0][0] for pos in positions]
    combined = ''.join(most_common_chars)
    return clean_ocr_number(combined)

def clean_ocr_number(s: str) -> Optional[float]:
    replacements = {
        'O': '0', 'o': '0',
        'I': '1', 'l': '1',
        'S': '5', 's': '5',
        'Z': '2',
        'D': '0',
        ',': '.',  # comma as decimal
    }

    cleaned = ''.join(replacements.get(c, c) for c in s if c.isalnum() or c in ['.', ','])

    try:
        return float(cleaned)
    except ValueError:
        return None

if __name__ == "__main__":
    # Example usage
    estimate_focal_length_text()
