#!/bin/bash

usage()
{
    echo 'Usage: ./convert.sh source_dir target_dir'
}

convert()
{
    if [ -d $1 ]; then
        mkdir -p $2
        for f in $1/*.mp3
        do
            echo "Converting " $f
            file=$(basename $f)
            filename="${file%.*}"
            ffmpeg -loglevel panic -i $f -vn -acodec pcm_s16le -ac 1 -ar 44100 -f wav $2/$filename".wav"
        done
    else
        echo "Invalid source_dir"
        usage
    fi
}

if [ $# == 2 ]; then
    convert $1 $2
else
    usage
fi
