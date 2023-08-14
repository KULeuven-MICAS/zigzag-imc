from __future__ import print_function

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import pandas as pd
import copy


def get_values(spreadsheet_id, range_name):
    """
    Creates the batch_update the user has access to.
    Load pre-authorized user credentials from the environment.
    TODO(developer) - See https://developers.google.com/identity
    for guides on implementing OAuth2 for the application.
        """
    #creds, _ = google.auth.default()
    # pylint: disable=maybe-no-member
    try:
        service = build('sheets', 'v4', developerKey='AIzaSyAI4GgFLWtp8EV29EYOE8b8STJAXkCyvP8')

        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id, range=range_name, majorDimension='COLUMNS').execute()
        rows = result.get('values', [])
        print(f"{len(rows[0])} rows retrieved")

        table = {}
        max_rows = 0
        for i, r in enumerate(rows):
            if len(r[1:]) > max_rows:
                max_rows = len(r[1:])

        for i, r in enumerate(rows):
            data_rows = copy.deepcopy(r)
            if len(r[1:]) < max_rows:
                data_rows += ['']*(max_rows - len(r[1:]))
            table[r[0]] = data_rows[1:]
        df = pd.DataFrame(table)
       
        return df
    except HttpError as error:
        print(f"An error occurred: {error}")
        return error


if __name__ == '__main__':
    # Pass: spreadsheet_id, and range_name
    table = get_values("14sOT8JJPU9aHC3rwLHjFkLKOu9jYdaPon6qBTLBuOeU", "AIMC!A1:AK100")
    

