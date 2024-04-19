import cProfile
import pstats
import main

# Exécuter le profilage sur le script
# cProfile.run('result.main()')

with cProfile.Profile() as profile:
    main.main()

results = pstats.Stats(profile)
results.sort_stats(pstats.SortKey.TIME)
results.print_stats()


