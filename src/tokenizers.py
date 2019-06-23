class Tokenizer:
  """
  Mapping ids to ints.
  """
  
  def fit(self, data):
    self.vocabulary = set(data)
    self.mapping = dict(zip(self.vocabulary, range(1, len(self.vocabulary) + 1)))
    return self
    
  def transform(self, data):
    if isinstance(data, list):
      return [self.mapping.get(d, 0) for d in data]
    else:
      return self.mapping.get(data, 0) 

  def get_all_tokenizers(train, item_metadata, filters):
    item_tokenizer = Tokenizer().fit(item_metadata['item_id'])
    user_tokenizer = Tokenizer().fit(train['user_id'])
    session_tokenizer = Tokenizer().fit(train['session_id'])

    action_type_tokenizer = Tokenizer().fit(train['action_type'])
    platform_tokenizer = Tokenizer().fit(train['platform'])
    city_tokenizer = Tokenizer().fit(train['city'])
    device_tokenizer = Tokenizer().fit(train['device'])
    filters_tokenizer = Tokenizer().fit(filters)

    tokenizers = {
      'item_id': item_tokenizer,
      'user_id': user_tokenizer,
      'session_id': session_tokenizer,
      'action_type': action_type_tokenizer,
      'platform': platform_tokenizer,
      'city': city_tokenizer,
      'device': device_tokenizer,
      'filters': filters_tokenizer
    }
    return tokenizers