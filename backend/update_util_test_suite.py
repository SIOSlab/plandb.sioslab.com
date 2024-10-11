import unittest
from update_util import *

df_original = pd.DataFrame({
  'pl_name': ['11 Com b','11 UMi b', '14 And b'], 
  'pl_letter': ['b', 'b', 'b'],
  'pl_orbper': [326.03, 516.22, 185.84]
  })

df_let_modified = pd.DataFrame({
  'pl_name': ['11 Com b','11 UMi b', '14 And b'], 
  'pl_letter': ['b', 'a', 'b'],
  'pl_orbper': [326.03, 516.22, 185.84]
  })

df_num_modified = pd.DataFrame({
  'pl_name': ['11 Com b','11 UMi b', '14 And b'], 
  'pl_letter': ['b','b', 'b'],
  'pl_orbper': [326.03, 516.22, 180.84]
  })

df_new_rows = pd.DataFrame({
  'pl_name': ['11 Com b','11 UMi b', '14 And b', '14 Her b'], 
  'pl_letter': ['b','b', 'b', 'b'],
  'pl_orbper': [326.03, 516.22, 185.84, 1773.4]
  })

df_modified_and_new_rows = pd.DataFrame({
  'pl_name': ['11 Com b','11 UMi b', '14 And b', '14 Her b'], 
  'pl_letter': ['b','b', 'b', 'b'],
  'pl_orbper': [326.03, 512.22, 185.84, 1773.4]
  })

df_let_changed_res = pd.DataFrame({
  'pl_name': ['11 UMi b'],
  'pl_letter': ['a'],
  'pl_orbper': [516.22]
})

df_num_changed_res = pd.DataFrame({
  'pl_name': ['14 And b'],
  'pl_letter': ['b'],
  'pl_orbper': [180.84]
})

df_new_res = pd.DataFrame({
  'pl_name': ['14 Her b'],
  'pl_letter': ['b'],
  'pl_orbper': [1773.4]
})


df_modified_new_res = pd.DataFrame({
  'pl_name': ['11 UMi b', '14 Her b'],
  'pl_letter': ['b', 'b'],
  'pl_orbper': [512.22, 1773.4]
})



# TODO: Mismatching col needed



class TestFunctions(unittest.TestCase):
  def test_get_ipac_differences(self):
    
    let_modified_differences, _ = get_ipac_differences(df_original, df_let_modified)
    num_modified_differences, _ = get_ipac_differences(df_original, df_num_modified)
    new_rows_differences, _ = get_ipac_differences(df_original, df_new_rows)
    modified_new_differences, _ = get_ipac_differences(df_original, df_modified_and_new_rows)
    
    pd.testing.assert_frame_equal(let_modified_differences, df_let_changed_res)
    pd.testing.assert_frame_equal(num_modified_differences, df_num_changed_res)
    pd.testing.assert_frame_equal(new_rows_differences, df_new_res)
    pd.testing.assert_frame_equal(modified_new_differences, df_modified_new_res)
    
  def test_upsert(self):
        
    df_merged_original = pd.DataFrame({
      'pl_name': ['11 Com b','11 UMi b', '14 And b'], 
      'pl_letter': ['b', 'b', 'b'],
      'pl_orbper': [326.03, 516.22, 185.84]
    })
    
    df_all_new = pd.DataFrame({
      'pl_name': ['14 Her b'], 
      'pl_letter': ['b'],
      'pl_orbper': [1773.4]
    })
    
    df_some_new = pd.DataFrame({
      'pl_name': ['11 Com b','11 UMi b', '14 And b', '14 Her b'], 
      'pl_letter': ['b', 'b', 'b', 'b'],
      'pl_orbper': [326.03, 516.22, 185.84, 1773.4]
    })
    
    df_updated = pd.DataFrame({
      'pl_name': ['11 Com b','11 UMi b', '14 And b'], 
      'pl_letter': ['b', 'b', 'b'],
      'pl_orbper': [326.03, 512.0, 185.84]
    })
    
    df_updated_new = pd.DataFrame({
      'pl_name': ['11 Com b','11 UMi b', '14 And b', '14 Her b'], 
      'pl_letter': ['b', 'b', 'b', 'b'],
      'pl_orbper': [326.03, 512, 185.84, 1773.4]
    })
    
    df_merged_all_res = pd.DataFrame({
      'pl_name': ['11 Com b','11 UMi b', '14 And b', '14 Her b'], 
      'pl_letter': ['b', 'b', 'b', 'b'],
      'pl_orbper': [326.03, 516.22, 185.84, 1773.4]
    })
    
    df_merged_some_res = pd.DataFrame({
      'pl_name': ['11 Com b','11 UMi b', '14 And b', '14 Her b'], 
      'pl_letter': ['b', 'b', 'b', 'b'],
      'pl_orbper': [326.03, 516.22, 185.84, 1773.4]
    })
    
    df_merged_updated_res = pd.DataFrame({
      'pl_name': ['11 Com b','11 UMi b', '14 And b'], 
      'pl_letter': ['b', 'b', 'b'],
      'pl_orbper': [326.03, 512, 185.84]
    })
    
    df_merged_updated_new_res = pd.DataFrame({
      'pl_name': ['11 Com b','11 UMi b', '14 And b', '14 Her b'], 
      'pl_letter': ['b', 'b', 'b', 'b'],
      'pl_orbper': [326.03, 512, 185.84, 1773.4]
    })
    
    print(df_some_new.columns)
    
    upsert_same = upsert_general(df_merged_original, df_merged_original, 'pl_name')
    upsert_all_new = upsert_general(df_merged_original, df_all_new, 'pl_name')
    upsert_some_new = upsert_general(df_merged_original, df_some_new, 'pl_name')
    upsert_updated = upsert_general(df_merged_original, df_updated, 'pl_name')
    upsert_updated_new = upsert_general(df_merged_original, df_updated_new, 'pl_name')
    
    
    
    pd.testing.assert_frame_equal(df_merged_original, upsert_same)
    pd.testing.assert_frame_equal(upsert_all_new, df_merged_all_res)
    pd.testing.assert_frame_equal(upsert_some_new, df_merged_some_res)
    pd.testing.assert_frame_equal(upsert_updated, df_merged_updated_res)
    pd.testing.assert_frame_equal(upsert_updated_new, df_merged_updated_new_res)
    
  def test_completeness_upsert(self):
      df_merged_original = pd.DataFrame({
        'completeness_id': [0, 1, 2, 3],
        'pl_id': [3, 3, 3, 3],
        'completeness': [0, 0, 0, 0.085041],
        'scenario_name': ['Conservative_NF_Imager_25hr', 'Conservative_NF_Imager_100hr', 'Conservative_NF_Imager_10000hr', 'Optimistic_NF_Imager_25hr'],
        'compMinWA': [None, None, None, None],
        'compMaxWA': [None, None, None, None],
        'compMindMag': [None, None, None, None],
        'compMaxdMag': [None, None, None, None]
      })
      
      df_merged_modified = pd.DataFrame({
        'completeness_id': [0, 1, 2, 3],
        'pl_id': [3, 3, 3, 3],
        'completeness': [0, 0, 0.084, 0.085041],
        'scenario_name': ['Conservative_NF_Imager_25hr', 'Conservative_NF_Imager_100hr', 'Conservative_NF_Imager_10000hr', 'Optimistic_NF_Imager_25hr'],
        'compMinWA': [None, None, None, None],
        'compMaxWA': [None, None, None, None],
        'compMindMag': [None, None, None, None],
        'compMaxdMag': [None, None, None, None]
      })
      
      df_merged_modified_new = pd.DataFrame({
        'completeness_id': [0, 1, 2, 3, 4],
        'pl_id': [3, 3, 3, 3, 3],
        'completeness': [0, 0, 0.084, 0.085041, 0.226834],
        'scenario_name': ['Conservative_NF_Imager_25hr', 'Conservative_NF_Imager_100hr', 'Conservative_NF_Imager_10000hr', 'Optimistic_NF_Imager_25hr', 'Optimistic_NF_Imager_100hr'],
        'compMinWA': [None, None, None, None, None],
        'compMaxWA': [None, None, None, None, None],
        'compMindMag': [None, None, None, None, None],
        'compMaxdMag': [None, None, None, None, None]
      })
      
      df_merged_new = pd.DataFrame({
        'completeness_id': [0, 1, 2, 3, 4],
        'pl_id': [3, 3, 3, 3, 3],
        'completeness': [0, 0, 0, 0.085041, 0.226834],
        'scenario_name': ['Conservative_NF_Imager_25hr', 'Conservative_NF_Imager_100hr', 'Conservative_NF_Imager_10000hr', 'Optimistic_NF_Imager_25hr', 'Optimistic_NF_Imager_100hr'],
        'compMinWA': [None, None, None, None, None],
        'compMaxWA': [None, None, None, None, None],
        'compMindMag': [None, None, None, None, None],
        'compMaxdMag': [None, None, None, None, None]
      })
      
      df_merged_single_new = pd.DataFrame({
        'completeness_id': [0],
        'pl_id': [3],
        'completeness': ['0.270857'],
        'scenario_name': ['Optimistic_NF_Imager_10000hr'],
        'compMinWA': [None],
        'compMaxWA': [None],
        'compMindMag': [None],
        'compMaxdMag': [None]
      })
      
      df_merged_modified_res = pd.DataFrame({
        'completeness_id': [0, 1, 2, 3],
        'pl_id': [3, 3, 3, 3],
        'completeness': [0, 0, 0.084, 0.085041],
        'scenario_name': ['Conservative_NF_Imager_25hr', 'Conservative_NF_Imager_100hr', 'Conservative_NF_Imager_10000hr', 'Optimistic_NF_Imager_25hr'],
        'compMinWA': [None, None, None, None],
        'compMaxWA': [None, None, None, None],
        'compMindMag': [None, None, None, None],
        'compMaxdMag': [None, None, None, None]
      })
      
      df_merged_modified_new_res = pd.DataFrame({
        'completeness_id': [0, 1, 2, 3, 4],
        'pl_id': [3, 3, 3, 3, 3],
        'completeness': [0, 0, 0.084, 0.085041, 0.226834],
        'scenario_name': ['Conservative_NF_Imager_25hr', 'Conservative_NF_Imager_100hr', 'Conservative_NF_Imager_10000hr', 'Optimistic_NF_Imager_25hr', 'Optimistic_NF_Imager_100hr'],
        'compMinWA': [None, None, None, None, None],
        'compMaxWA': [None, None, None, None, None],
        'compMindMag': [None, None, None, None, None],
        'compMaxdMag': [None, None, None, None, None]
      })
      
      df_merged_new_res = pd.DataFrame({
        'completeness_id': [0, 1, 2, 3, 4],
        'pl_id': [3, 3, 3, 3, 3],
        'completeness': [0, 0, 0, 0.085041, 0.226834],
        'scenario_name': ['Conservative_NF_Imager_25hr', 'Conservative_NF_Imager_100hr', 'Conservative_NF_Imager_10000hr', 'Optimistic_NF_Imager_25hr', 'Optimistic_NF_Imager_100hr'],
        'compMinWA': [None, None, None, None, None],
        'compMaxWA': [None, None, None, None, None],
        'compMindMag': [None, None, None, None, None],
        'compMaxdMag': [None, None, None, None, None]
      })
      
      df_merged_single_res = pd.DataFrame({
        'completeness_id': [0],
        'pl_id': [3],
        'completeness': ['0.270857'],
        'scenario_name': ['Optimistic_NF_Imager_10000hr'],
        'compMinWA': [None],
        'compMaxWA': [None],
        'compMindMag': [None],
        'compMaxdMag': [None]
      })
      
      upsert_same = upsert_general(df_merged_original, df_merged_original, 'pl_id')
      upsert_modified = upsert_general(df_merged_original, df_merged_modified, 'pl_id')
      upsert_modified_new = upsert_general(df_merged_original, df_merged_modified_new, 'pl_id')
      upsert_new = upsert_general(df_merged_original, df_merged_new, 'pl_id')
      upsert_single = upsert_general(df_merged_original, df_merged_single_new, 'pl_id')
      
      print(upsert_modified_new)
      print(df_merged_modified_new_res)
      
      
      pd.testing.assert_frame_equal(df_merged_original, upsert_same)
      pd.testing.assert_frame_equal(upsert_modified, df_merged_modified_res)
      pd.testing.assert_frame_equal(upsert_modified_new, df_merged_modified_new_res)
      pd.testing.assert_frame_equal(upsert_new, df_merged_new_res)
      pd.testing.assert_frame_equal(upsert_single, df_merged_single_res)
      
      return
    
  def test_upsert(self):
    
    
    
    
    return
    
    
if __name__ == '__main__':
  unittest.main()