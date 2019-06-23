import numpy as np


def gen_properties_impressions(row, item_metadata, properties, SIZE):
    impressions = row['impressions'].split('|')

    properties_local = []
    for impression in impressions:
        if impression in item_metadata.index:
            properties_local.append(item_metadata.loc[int(impression)]['properties'])
        else:
            properties_local.append('')

    vector = np.zeros((25, SIZE))
    for i, pr in enumerate(properties_local):
      prs = pr.split('|')
      for p in prs:
          if p in properties:
            vector[i, properties[p]] = 1
     
    return vector.reshape(-1).tolist()

