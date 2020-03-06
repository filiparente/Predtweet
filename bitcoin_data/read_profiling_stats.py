import pstats
p = pstats.Stats(r'C:\Users\Filipa\Desktop\Predtweet\bitcoin_data\profiling_stats.txt')
p.sort_stats('tottime').print_stats(20)