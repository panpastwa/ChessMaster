# This script takes as input pgn file downloaded from lichess database and creates new file
# containing all positions that occured in games saved in FEN format
# NOTE: You need to have pgn-extract installed and added to PATH

# USAGE: bash prepare_fens.sh input_file output_file
pgn-extract $1 --output $2 -Wfen --notags
