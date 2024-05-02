import cProfile
import pstats
import testprofiling

# Exécuter le profilage sur le script
# cProfile.run('result.main()')

with cProfile.Profile() as profile:
    testprofiling

results = pstats.Stats(profile)
results.sort_stats(pstats.SortKey.TIME)
results.print_stats()


