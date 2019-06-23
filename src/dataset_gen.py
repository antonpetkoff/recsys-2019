import pandas as pd
import tensorflow as tf
from src.utils import gen_properties_impressions
gen_properties_impressions
ITEM_ACTIONS = [
 #'interaction item deals',
 'interaction item image',
 #'interaction item info',
 #'interaction item rating',
]

class InputGenerator:
  all_fields = ['user_id', 'session_id', 'platform', 'city',
                'device','action_types', 'interacted_items',
                'filters', 'items', 'prices', 'timestamp', 'impressions_filters']
  simple_fields = ['user_id', 'session_id', 'platform', 'city', 'device']
  
  def __init__(self, tokenizers, returned_fields=None):
    self.tokenizers = tokenizers
    self.returned_fields = returned_fields
    print(self.returned_fields)
    num_fields = len(returned_fields) if returned_fields else len(self.all_fields)
    self.output_types = list((tf.int32 for _ in range(num_fields)))
    if 'prices' in returned_fields:
      self.output_types[returned_fields.index('prices')] = tf.float32
    self.output_types = tuple(self.output_types)
    self.padded_shapes = tuple([tf.TensorShape([None]) for _ in range(num_fields)])
    
    
  def prepare_last_row(self, row): 
    # Dealing with current_filters	impressions	prices
    impressions = list(map(int, row['impressions'].split('|')))
    items = self.tokenizers['item_id'].transform(impressions)
    prices = list(map(int, row['prices'].split('|')))
    clicked_item = self.tokenizers['item_id'].transform(int(row['reference']))

    return items, prices, clicked_item

  def input_generator_gen(self, df, return_items_filters=False, item_metadata=None, properties=None, return_idx=False):
    """
      Returns input generator, types and shapes used
      by the TF dataset.
    """
    def gen():
      last_step = None
      session_id = None
      size = len(properties)

      for index, row in df.iterrows():
        user_id, new_session_id, platform, city, device = [self.tokenizers[field].transform(row[field]) for field in self.simple_fields]

        if session_id != new_session_id:
          # Reinitialize everything because a new session started.
          session_id = new_session_id
          action_types, interacted_items = [], []
          filters = set()
          last_step = 0
       
        last_step += 1
        # assert row['step'] == last_step
        
        if not pd.isnull(row['current_filters']):
          filters.update(self.tokenizers['filters'].transform(row['current_filters'].split('|')))

        if row['action_type'] in ITEM_ACTIONS and not pd.isnull(row['reference']):
          interacted_item = self.tokenizers['item_id'].transform(int(row['reference']))
          interacted_items.append(interacted_item)
        
        action_types.append(self.tokenizers['action_type'].transform(row['action_type']))

        # Checking if it is the last row from the session
        if row['action_type'] == 'clickout item':
          new_session = True
          items, prices, clicked_item = self.prepare_last_row(row)

          impressions_filters = []
          if return_items_filters:
            impressions_filters = gen_properties_impressions(row,
                                                                   item_metadata,
                                                                   properties,
                                                                   size)

          result = {
              'user_id': [user_id],
              'session_id': [session_id],
              'platform': [platform],
              'city': [city],
              'device': [device],
              'action_types': action_types,
              'interacted_items': interacted_items,
              'filters': list(filters),
              'items': items,
              'prices': prices,
              'timestamp': [row['timestamp']],
              'impressions_filters': impressions_filters
          }
          
          if self.returned_fields:
            returned_values = [result[f] for f in self.returned_fields]
          else:
            returned_values = result.values()
          
          if return_idx:
            clicked_item = result['items'].index(clicked_item) if clicked_item in result['items'] else 0
          yield tuple(returned_values), [clicked_item]
          interacted_items = []


    return gen, (self.output_types, tf.int32), (self.padded_shapes, tf.TensorShape([None]))
