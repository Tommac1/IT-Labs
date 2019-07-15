#!/bin/bash

#############################
# FUNCTIONS
#############################

# usage info
show_help() {
cat << EOF
usage: ${0} -a dir -b dir -c [lsr|per] [OPT...]
    -i filename  select one image (Aerial.bmp, ...)
    -c [lsr | per] select coeff (Lsr = [0.1 - 1.5] or percent = [10% - 100%])
    -x val       pick specific coeff val (i.e. 0.5 for Lsr or 50% for percent)
    -a dir/aft   after compression dir
    -b dir/bef   before compression dir
    -p           make psnr
    -s           make ssim
    -e           make entrophy
    -d dir/docs  make docs (csv files for ssim, psnr and entr)
    -h           show help

examples:
    $0 -b img/before -a img/after -c lsr -p -s -e -d docs
    $0 -b img/before -a img/after -c per -x 50 -i Aerial.bmp -s -p -d docs
    $0 -b img/before -a img/after -c lsr -d docs
EOF
}

process_psnr() {
echo "================================="
echo "PSNR"

for FILE in $IMAGES; do
    echo $FILE
    FILE_BASE=${FILE%.bmp}
    mkdir -p "$METRICS_DIR/$FILE_BASE"
    : > "$METRICS_DIR/$FILE_BASE/psnr"

    for COEFF in $COEFFS; do
        echo -n "$COEFF "
        FILE_OUTPUT=$FILE_BASE$COEFF

        echo $FILE_OUTPUT >> "$METRICS_DIR/$FILE_BASE/psnr"
        ./psnr "$BEFORE_DIR/$FILE" "$AFTER_DIR/$COEFF/$FILE" \
                "$METRICS_DIR/$FILE_BASE/$FILE_OUTPUT" \
                >> "$METRICS_DIR/$FILE_BASE/psnr"
    done
    echo ""
done
}

process_ssim() {
echo "================================="
echo "SSIM"

for FILE in $IMAGES; do
    echo $FILE
    FILE_BASE=${FILE%.bmp}
    mkdir -p "$METRICS_DIR/$FILE_BASE"
    : > "$METRICS_DIR/$FILE_BASE/ssim"

    for COEFF in $COEFFS; do
        echo -n "$COEFF "
        FILE_OUTPUT=$FILE_BASE$COEFF

        echo $FILE_OUTPUT >> "$METRICS_DIR/$FILE_BASE/ssim"
        ./compute_ssim.sh "$BEFORE_DIR/$FILE" \
                "$AFTER_DIR/$COEFF/$FILE" \
                >> "$METRICS_DIR/$FILE_BASE/ssim"
    done
    echo ""
done
}

adjust_coeff() {
# for jp2 coeffs we have to change coeffs from 02 -> 0.2 etc.
    TMP="${1:0:1}.${1:1:1}"
    echo $TMP
}

process_entrophy() {
echo "================================="
echo "ENTROPHY"

if [[ -z $ENT_IMAGES ]]; then
    ENT_IMAGES=$(ls $METRICS_DIR)
fi

for ENT_IMAGE in $ENT_IMAGES; do
    echo $ENT_IMAGE
    ENT_IMAGE_BASE=${ENT_IMAGE%.bmp}

    : > "$METRICS_DIR/$ENT_IMAGE_BASE/entr"

    for COEFF in $COEFFS; do
        echo -n "$COEFF "
        FILE_OUTPUT="${ENT_IMAGE_BASE}${COEFF}_error"

        if [[ "lsr" == $COEFF_TYPE ]]; then
            # 01 -> 0.1; 02 -> 0.2, etc...
            COEFF=$(adjust_coeff $COEFF)
        fi

        # inverted h and w because $identify is misplacing them (or do we?)
        IMAGE_SIZE_W=$(identify -format "%h" "$BEFORE_DIR/$ENT_IMAGE")
        IMAGE_SIZE_H=$(identify -format "%w" "$BEFORE_DIR/$ENT_IMAGE")
        ./entrophy "$METRICS_DIR/$ENT_IMAGE_BASE/$FILE_OUTPUT" \
                $IMAGE_SIZE_H $IMAGE_SIZE_W $COEFF        \
                >> "$METRICS_DIR/$ENT_IMAGE_BASE/entr"
    done
    echo ""
done
}

make_csv() {
echo "================================="
echo "CSV FILES"

mkdir -p $DOCS_DIR

: > "$PSNR_CSV_FILE"
: > "$SSIM_CSV_FILE"
: > "$ENTR_CSV_FILE"

TMP=$(echo $COEFFS | tr ' ' ',')
PSNR_TMP=$(echo $TMP | tr ',' ',,')
echo "file_name,$PSNR_TMP" >> $PSNR_CSV_FILE
echo "file_name,$TMP" >> $SSIM_CSV_FILE
echo "file_name,$TMP" >> $ENTR_CSV_FILE

for FILE in $IMAGES; do
    echo -n "$FILE," >> $PSNR_CSV_FILE
    echo -n "$FILE," >> $SSIM_CSV_FILE
    echo -n "$FILE," >> $ENTR_CSV_FILE

    FILE_BASE=${FILE%.bmp}
    PSNR_FILE="$METRICS_DIR/$FILE_BASE/psnr"
    SSIM_FILE="$METRICS_DIR/$FILE_BASE/ssim"
    ENTR_FILE="$METRICS_DIR/$FILE_BASE/entr"

    [ -f $PSNR_FILE ] && cat $PSNR_FILE | awk '{ print $3 }' \
            | awk 'NF' | tr '\n' ',' >> $PSNR_CSV_FILE
    echo >> $PSNR_CSV_FILE

    [ -f $SSIM_FILE ] && cat $SSIM_FILE | grep ans | awk '{ print $3 }' \
            | awk 'NF' | tr '\n' ',' >> $SSIM_CSV_FILE
    echo >> $SSIM_CSV_FILE

    [ -f $ENTR_FILE ] && cat $ENTR_FILE | grep Lsr | awk '{ print $3 }' \
            | awk 'NF' | tr '\n' ',' >> $ENTR_CSV_FILE
    echo >> $ENTR_CSV_FILE
done
}

#############################
# MAIN
#############################

PSNR_CSV_FILE="psnr.csv"
SSIM_CSV_FILE="ssim.csv"
ENTR_CSV_FILE="entr.csv"

METRICS_DIR="metrics"
DOCS_DIR=""

LSR_COEFFS="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15"
PER_COEFFS="10 20 30 40 50 60 70 80 90 100"
COEFFS=""
COEFF_TYPE=""
IMAGES=""
ENT_IMAGES=""

MAKE_PSNR=0
MAKE_SSIM=0
MAKE_CSV=0
MAKE_ENTR=0

# ARGUMENTS (OPTS) PARSE
# posix var reset
OPTIND=1
while getopts "i:c:a:b:d:x:pseh?" opt; do
    case "$opt" in
    i )  IMAGES=$OPTARG
         ENT_IMAGES=$OPTARG
        ;;
    c )  COEFF_TYPE=$OPTARG
        ;;
    x )  COEFFS=$(echo $OPTARG | tr -d '%')
        ;;
    a )  AFTER_DIR=$OPTARG
        ;;
    b )  BEFORE_DIR=$OPTARG
        ;;
    p )  MAKE_PSNR=1
        ;;
    s )  MAKE_SSIM=1
        ;;
    e )  MAKE_ENTR=1
        ;;
    d )  MAKE_CSV=1
         DOCS_DIR=$OPTARG
        ;;
    : )  echo "$OPTARG requres argument"
         exit 1
        ;;
    h|\? )
        show_help
        exit 1
        ;;
    esac
done

make all
RESULT=$?


# compilation failed
if [[ $RESULT -ne 0 ]]; then
    exit # POINT OF NO RETURN
fi

if [[ -z $AFTER_DIR ]] || [[ -z $BEFORE_DIR ]] \
    || [[ -z $COEFF_TYPE ]]; then
    show_help
    exit 1
fi

# if image is not specified - take whole set
if [[ -z $IMAGES ]]; then
    IMAGES=$(ls $BEFORE_DIR)
fi

# choose coefficients if not specified
if [[ -z $COEFFS ]]; then
    if [[ "lsr" == $COEFF_TYPE ]]; then
        COEFFS=$LSR_COEFFS
    else
        COEFFS=$PER_COEFFS
    fi
fi

METRICS_DIR="$METRICS_DIR/$AFTER_DIR"
# docs dir specified - make relative path
if [[ -n $DOCS_DIR ]]; then
    PSNR_CSV_FILE="$DOCS_DIR/$PSNR_CSV_FILE"
    SSIM_CSV_FILE="$DOCS_DIR/$SSIM_CSV_FILE"
    ENTR_CSV_FILE="$DOCS_DIR/$ENTR_CSV_FILE"
fi

###################
# PROCESS PSNR
if [[ $MAKE_PSNR -eq 1 ]]; then
    process_psnr
fi

###################
# PROCESS SSIM
if [[ $MAKE_SSIM -eq 1 ]]; then
    process_ssim
fi

###################
# PROCESS ENTROPHY
if [[ $MAKE_ENTR -eq 1 ]]; then
    process_entrophy
fi

###################
# PROCESS CSV

# remove csv files with metrics
[ -e $PSNR_CSV_FILE ] && rm $PSNR_CSV_FILE
[ -e $SSIM_CSV_FILE ] && rm $SSIM_CSV_FILE
[ -e $ENTR_CSV_FILE ] && rm $ENTR_CSV_FILE

if [[ $MAKE_CSV -eq 1 ]]; then
    make_csv
fi
