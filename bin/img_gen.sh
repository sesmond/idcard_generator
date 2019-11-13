if [ "$4" = "" ]; then
    echo "Usage: img_gen.sh <type:train|test|validate> <dir:data> <num:100> <worker>"
    exit
fi

python -m idcardgenerator --type=$1 --dir=$2 --num=$3 --worker=$4
