## DAM microstructure feature sanity

**Model eğitimi yapılmadı.**

### Master doluluk
- `dam_bid_volume_mwh` missing=0 inf=0 p01=16882.9 p99=37674
- `dam_sell_offer_volume_mwh` missing=0 inf=0 p01=17712.9 p99=44303.5
- `dam_matched_buy_mwh` missing=0 inf=0 p01=14356.4 p99=32981.9
- `dam_matched_sell_mwh` missing=0 inf=0 p01=14356.3 p99=32982
- `dam_block_matched_buy_mwh` missing=0 inf=0 p01=3.1 p99=3756.3
- `dam_block_unmatched_buy_mwh` missing=0 inf=0 p01=0 p99=4170.54

### Feature parquet (split bazında özet)
#### split=test
- `dam_bid_volume_mwh` missing=0 inf=0 min=10518.3 max=37831.9
- `dam_sell_offer_volume_mwh` missing=0 inf=0 min=18561.0 max=51072.2
- `dam_matched_buy_mwh` missing=0 inf=0 min=10518.3 max=32624.2
- `dam_matched_sell_mwh` missing=0 inf=0 min=10517.8 max=32624.2
- `dam_block_matched_buy_mwh` missing=0 inf=0 min=96.5 max=4075.3
- `dam_block_unmatched_buy_mwh` missing=0 inf=0 min=0.0 max=2553.2
- `dam_matched_volume_mwh` missing=0 inf=0 min=10518.05 max=32624.2
- `dam_bid_to_match_ratio` missing=0 inf=0 min=0.9999231226759784 max=1.4315558313839691
- `dam_sell_to_match_ratio` missing=0 inf=0 min=0.9999827947301823 max=4.126918340172782
- `dam_unmatched_buy_proxy` missing=0 inf=0 min=0.0 max=11299.400000000001
- `dam_unmatched_sell_proxy` missing=0 inf=0 min=0.0 max=34787.3
- `dam_block_total_buy_mwh` missing=0 inf=0 min=142.89999999999998 max=4756.3
- `dam_block_unmatched_ratio` missing=0 inf=0 min=0.0 max=0.9474191992281717
- `dam_block_pressure` missing=0 inf=0 min=142.89999999999998 max=4756.3
- `dam_buy_sell_ratio` missing=0 inf=0 min=0.24231045373058063 max=1.3884629061652665
- `dam_offer_supply_demand_gap` missing=0 inf=0 min=-34788.3 max=10487.099999999999
- `dam_offer_total_volume_mwh` missing=0 inf=0 min=34262.9 max=80157.3
- `dam_offer_balance_pressure` missing=0 inf=0 min=-0.609903542221773 max=0.16264138126765082
- `dam_match_ratio` missing=0 inf=0 min=0.19504911528663432 max=0.4742978953119309
- `dam_bid_volume_lag_24` missing=0 inf=0 min=10518.3 max=37831.9
- `dam_sell_offer_volume_lag_24` missing=0 inf=0 min=18561.0 max=51072.2
- `dam_buy_sell_ratio_lag_24` missing=0 inf=0 min=0.24231045373058063 max=1.3884629061652665
- `dam_offer_balance_pressure_lag_24` missing=0 inf=0 min=-0.609903542221773 max=0.16264138126765082
- `dam_match_ratio_lag_24` missing=0 inf=0 min=0.19504911528663432 max=0.4742978953119309
- `dam_block_unmatched_ratio_lag_24` missing=0 inf=0 min=0.0 max=0.9474191992281717
- `dam_bid_volume_lag_168` missing=0 inf=0 min=11984.9 max=37831.9
- `dam_sell_offer_volume_lag_168` missing=0 inf=0 min=23644.0 max=51072.2
- `dam_buy_sell_ratio_lag_168` missing=0 inf=0 min=0.29355681920238985 max=1.3884629061652665
- `dam_offer_balance_pressure_lag_168` missing=0 inf=0 min=-0.5461245847965184 max=0.16264138126765082

#### split=train
- `dam_bid_volume_mwh` missing=0 inf=0 min=13663.9 max=47542.8
- `dam_sell_offer_volume_mwh` missing=0 inf=0 min=10444.2 max=49452.1
- `dam_matched_buy_mwh` missing=0 inf=0 min=10443.0 max=34975.3
- `dam_matched_sell_mwh` missing=0 inf=0 min=10444.2 max=34975.3
- `dam_block_matched_buy_mwh` missing=0 inf=0 min=0.0 max=12389.499999999998
- `dam_block_unmatched_buy_mwh` missing=0 inf=0 min=0.0 max=11283.100000000002
- `dam_matched_volume_mwh` missing=0 inf=0 min=10443.6 max=34975.3
- `dam_bid_to_match_ratio` missing=0 inf=0 min=0.9998839365867408 max=2.9162511367580324
- `dam_sell_to_match_ratio` missing=0 inf=0 min=0.999893964975816 max=2.722814681238049
- `dam_unmatched_buy_proxy` missing=0 inf=0 min=0.0 max=30019.600000000002
- `dam_unmatched_sell_proxy` missing=0 inf=0 min=0.0 max=24903.100000000002
- `dam_block_total_buy_mwh` missing=0 inf=0 min=0.0 max=17137.000000000004
- `dam_block_unmatched_ratio` missing=0.00544872 inf=0 min=0.0 max=0.9999314810373772
- `dam_block_pressure` missing=0 inf=0 min=0.0 max=17137.000000000004
- `dam_buy_sell_ratio` missing=0 inf=0 min=0.37312554601983655 max=2.916187078804706
- `dam_offer_supply_demand_gap` missing=0 inf=0 min=-24789.299999999996 max=30006.9
- `dam_offer_total_volume_mwh` missing=0 inf=0 min=31945.6 max=85083.29999999999
- `dam_offer_balance_pressure` missing=0 inf=0 min=-0.45653105486037426 max=0.48929916785016364
- `dam_match_ratio` missing=0 inf=0 min=0.2553448070843794 max=0.4736149488420571
- `dam_bid_volume_lag_24` missing=0 inf=0 min=13663.9 max=47542.8
- `dam_sell_offer_volume_lag_24` missing=0 inf=0 min=10444.2 max=49452.1
- `dam_buy_sell_ratio_lag_24` missing=0 inf=0 min=0.37312554601983655 max=2.916187078804706
- `dam_offer_balance_pressure_lag_24` missing=0 inf=0 min=-0.45653105486037426 max=0.48929916785016364
- `dam_match_ratio_lag_24` missing=0 inf=0 min=0.2553448070843794 max=0.4736149488420571
- `dam_block_unmatched_ratio_lag_24` missing=0.00544872 inf=0 min=0.0 max=0.9999314810373772
- `dam_bid_volume_lag_168` missing=0 inf=0 min=13663.9 max=47542.8
- `dam_sell_offer_volume_lag_168` missing=0 inf=0 min=10444.2 max=49452.1
- `dam_buy_sell_ratio_lag_168` missing=0 inf=0 min=0.37312554601983655 max=2.916187078804706
- `dam_offer_balance_pressure_lag_168` missing=0 inf=0 min=-0.45653105486037426 max=0.48929916785016364

#### split=validation
- `dam_bid_volume_mwh` missing=0 inf=0 min=19097.0 max=41618.1
- `dam_sell_offer_volume_mwh` missing=0 inf=0 min=20702.8 max=48183.3
- `dam_matched_buy_mwh` missing=0 inf=0 min=16791.3 max=36934.2
- `dam_matched_sell_mwh` missing=0 inf=0 min=16791.3 max=36934.2
- `dam_block_matched_buy_mwh` missing=0 inf=0 min=91.89999999999999 max=5067.5
- `dam_block_unmatched_buy_mwh` missing=0 inf=0 min=0.0 max=4755.3
- `dam_matched_volume_mwh` missing=0 inf=0 min=16791.3 max=36934.2
- `dam_bid_to_match_ratio` missing=0 inf=0 min=0.9999320017770202 max=1.4382115959148307
- `dam_sell_to_match_ratio` missing=0 inf=0 min=0.9999752884252632 max=2.057811343974566
- `dam_unmatched_buy_proxy` missing=0 inf=0 min=0.0 max=11466.900000000001
- `dam_unmatched_sell_proxy` missing=0 inf=0 min=0.0 max=24694.500000000004
- `dam_block_total_buy_mwh` missing=0 inf=0 min=168.5 max=5885.300000000001
- `dam_block_unmatched_ratio` missing=0 inf=0 min=0.0 max=0.9720762893763158
- `dam_block_pressure` missing=0 inf=0 min=168.5 max=5885.300000000001
- `dam_buy_sell_ratio` missing=0 inf=0 min=0.48660890791260547 max=1.3842280995565668
- `dam_offer_supply_demand_gap` missing=0 inf=0 min=-24695.000000000004 max=11017.7
- `dam_offer_total_volume_mwh` missing=0 inf=0 min=42071.7 max=80125.4
- `dam_offer_balance_pressure` missing=0 inf=0 min=-0.3453437480125578 max=0.16115408573031573
- `dam_match_ratio` missing=0 inf=0 min=0.31435912655272724 max=0.47181069847635615
- `dam_bid_volume_lag_24` missing=0 inf=0 min=19097.0 max=41618.1
- `dam_sell_offer_volume_lag_24` missing=0 inf=0 min=20702.8 max=48183.3
- `dam_buy_sell_ratio_lag_24` missing=0 inf=0 min=0.48660890791260547 max=1.3842280995565668
- `dam_offer_balance_pressure_lag_24` missing=0 inf=0 min=-0.3453437480125578 max=0.16115408573031573
- `dam_match_ratio_lag_24` missing=0 inf=0 min=0.31435912655272724 max=0.47181069847635615
- `dam_block_unmatched_ratio_lag_24` missing=0 inf=0 min=0.0 max=0.9720762893763158
- `dam_bid_volume_lag_168` missing=0 inf=0 min=19097.0 max=41618.1
- `dam_sell_offer_volume_lag_168` missing=0 inf=0 min=20702.8 max=48183.3
- `dam_buy_sell_ratio_lag_168` missing=0 inf=0 min=0.48660890791260547 max=1.3842280995565668
- `dam_offer_balance_pressure_lag_168` missing=0 inf=0 min=-0.3453437480125578 max=0.16115408573031573

### Safe division / consistency checks
- **dam_buy_sell_ratio**: {'inf_count': 0}
- **dam_offer_balance_pressure**: {'inf_count': 0}
- **dam_match_ratio**: {'inf_count': 0}
- **dam_bid_to_match_ratio**: {'inf_count': 0}
- **dam_sell_to_match_ratio**: {'inf_count': 0}
- **dam_block_unmatched_ratio**: {'inf_count': 0}
- **matched_buy_sell_mismatch_rows**: 16274
- **matched_volume_avg_abs_diff_p99**: 0.0

### Target korelasyonları (overall)
- `dam_bid_volume_mwh` corr(target_1h)=0.36436485562714593 corr(target_24h)=0.30234248762522625
- `dam_sell_offer_volume_mwh` corr(target_1h)=-0.13623230682879675 corr(target_24h)=-0.12195788517391803
- `dam_matched_buy_mwh` corr(target_1h)=0.32385056723250305 corr(target_24h)=0.2722622340212553
- `dam_matched_sell_mwh` corr(target_1h)=0.32384159215158836 corr(target_24h)=0.27225424773901014
- `dam_block_matched_buy_mwh` corr(target_1h)=-0.06534743793557973 corr(target_24h)=-0.051944958690232704
- `dam_block_unmatched_buy_mwh` corr(target_1h)=-0.21074402947157136 corr(target_24h)=-0.2406258891275155
- `dam_matched_volume_mwh` corr(target_1h)=0.32384607987129166 corr(target_24h)=0.27225824103162427
- `dam_bid_to_match_ratio` corr(target_1h)=0.07100975447936804 corr(target_24h)=0.05351552033382417
- `dam_sell_to_match_ratio` corr(target_1h)=-0.4549146816519516 corr(target_24h)=-0.3910300432258907
- `dam_unmatched_buy_proxy` corr(target_1h)=0.17784330811024454 corr(target_24h)=0.13979911683169782
- `dam_unmatched_sell_proxy` corr(target_1h)=-0.41723649801218055 corr(target_24h)=-0.3590529995914637
- `dam_block_total_buy_mwh` corr(target_1h)=-0.203236159736601 corr(target_24h)=-0.21649198032266584
- `dam_block_unmatched_ratio` corr(target_1h)=0.015260120239033297 corr(target_24h)=-0.04062070855948793
- `dam_block_pressure` corr(target_1h)=-0.203236159736601 corr(target_24h)=-0.21649198032266584
- `dam_buy_sell_ratio` corr(target_1h)=0.3419477246497446 corr(target_24h)=0.2900708611230643
- `dam_offer_supply_demand_gap` corr(target_1h)=0.3824286259018234 corr(target_24h)=0.32519283878121374
- `dam_offer_total_volume_mwh` corr(target_1h)=0.1126889597489566 corr(target_24h)=0.08727565043213843
- `dam_offer_balance_pressure` corr(target_1h)=0.39260396064557596 corr(target_24h)=0.33357191220414567
- `dam_match_ratio` corr(target_1h)=0.4613645851141183 corr(target_24h)=0.39827829928004416
- `dam_bid_volume_lag_24` corr(target_1h)=0.2974951320222816 corr(target_24h)=0.2638425888599307
- `dam_sell_offer_volume_lag_24` corr(target_1h)=-0.10974087751179748 corr(target_24h)=-0.10959130314124246
- `dam_buy_sell_ratio_lag_24` corr(target_1h)=0.2774747554107 corr(target_24h)=0.2541728141939194
- `dam_offer_balance_pressure_lag_24` corr(target_1h)=0.3201214242074305 corr(target_24h)=0.2955458112521575
- `dam_match_ratio_lag_24` corr(target_1h)=0.3884305335979897 corr(target_24h)=0.3567508998463631
- `dam_block_unmatched_ratio_lag_24` corr(target_1h)=-0.04454420445378035 corr(target_24h)=-0.07470428967034697
- `dam_bid_volume_lag_168` corr(target_1h)=0.31401655969101877 corr(target_24h)=0.2683933173985509
- `dam_sell_offer_volume_lag_168` corr(target_1h)=-0.10221106793451626 corr(target_24h)=-0.11056822658056194
- `dam_buy_sell_ratio_lag_168` corr(target_1h)=0.2822223046535893 corr(target_24h)=0.2585361142727652
- `dam_offer_balance_pressure_lag_168` corr(target_1h)=0.3264012539393206 corr(target_24h)=0.2986822301079554

### Mod notu
- post_dam_publication_mode: current değerler kullanılabilir.
- strict_forecast_mode: lagged versiyonlar tercih edilmeli.