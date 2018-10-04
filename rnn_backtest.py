
#
# commission = 0.002
# commission = 0
# returns_from_strat = []
# col = []
#
# current_position = 0
# buy_signal = 0
# sell_signal = 0
# hold_time = 0
# for t, ret, pred in zip(test_df_2.index, test_df_2["target_pct"], test_df_2["pred"]):
#     if hold_time > 0:
#         returns_from_strat.append(ret)
#         hold_time = hold_time - 1
#     else:
#         returns_from_strat.append(0)
#
#     if pred:
#         hold_time = 3

#    if buy_signal:
#        returns_from_strat.append(ret - commission)
#        current_position = 1
#        hold_time = 3
#        buy_signal = 0
#    elif sell_signal:
#        returns_from_strat.append(-1*ret - commission)
#        current_position = -1
#        hold_time = 3
#        sell_signal = 0
#    else:
#        returns_from_strat.append(0)
#
#    if current_position == 0 and pred:
#        buy_signal = 1
#        hold_time = 3
#
#    if current_position != 0:
#        hold_time = hold_time - 1
#
#    if hold_time == 1:
#        returns_from_strat.append()



# test_df_2["ret_strat"] = (np.array(returns_from_strat) + 1).cumprod()

#(test_df_2["ret_strat"]*1).plot()
#(test_df_2["pred"]*1).plot()


#test_df_3 = test_df_2.iloc[0:200,:]
# import matplotlib
# plt.pyplot.scatter(x = test_df_2.index, y= test_df_2["ret_strat"], c = test_df_2["pred"])


#test_x.shape

#len(test_main_df)
#len(test_x)
#train_y = np.array(train_y).reshape(-1, 1)
#val_y = np.array(val_y).reshape(-1, 1)
#
#shape_x = np.array(train_x).shape
#shape_y = np.array(val_y).shape
#shape_y = np.array(train_y).shape
#
#
#train_x.shape
#np.array(y_batches).shape
