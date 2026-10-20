#!/bin/bash

usage() {
  echo -e "Usage: bash fragment2bw_cov.sh -i <input.fragments.tsv.gz> -o <output.bw> -c <genome..chrom.sizes> -t <INT> target sum> -h\n\n"
  echo -e "Description: Generates library size normalised coverage tracks in bigwig format from fragments.tsv.gz files that are usually output by ATAC seq pipelines (e.g. cellranger)\nNeeds environment with bedtools bedGraphToBigWig available (e.g. genomics_python).\n"
  echo -e "  -i/--input: Path to the input fragments.tsv.gz file (like produced by e.g. the cellranger pipeline). This file is assumed to have 1 line per ATAC fragment, the first 3 columns must be chrom start stop in 0-based coordinates. The file can have more columns but this information is discarded.\n"
  echo -e "  -o/--output_file: Path to output to be generated (Normalised bigwig file, must end with .bw)\n"
  echo -e "  -c/--chrom_sizes: Path to chrom.sizes file of the genome used, needed for bedtools genomecov (-g). E.g. Two-column tab-separated text file containing assembly
    sequence names and sizes as in https://hgdownload.cse.ucsc.edu/goldenpath/hg38/bigZips/\n"
  echo -e "  -h: Display this help message\n"
  exit 1
}

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -i|--input)
      fragments_input_file="$2"
      shift 2
      ;;
    -o|--output_file)
      output_file="$2"
      shift 2
      ;;
    -c|--chrom_sizes)
      chrom_sizes_file="$2"
      shift 2
      ;;
    -t|--target_sum)
          FRAG_TARGET_SUM="$2"
          shift 2
          ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Invalid option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

#file checks

#check if filenames have been provided and has correct ending
if [[ -z "$fragments_input_file" ]]; then
  echo "fragments input file not provided, exiting";
  exit 1;
fi
if [[ $fragments_input_file != *.tsv.gz ]];then 
    echo "incorrect ending: fragments file does not end on .tsv.gz:";
    echo $fragments_input_file;
    exit 1;
fi

#check if output prefix has been provided
if [[ -z "$output_file" ]]; then
  echo "output output file not provided, deriving from input name";
  output_prefix=$(basename ${fragments_input_file/.tsv/})
  output_dir=$(dirname ${fragments_input_file})
elif [[ $output_file != *.bw ]]; then
    echo "Output file must end with .bw, exiting";
    exit 1;
else
    output_prefix=$(basename ${output_file/.bw/})
    output_dir=$(dirname $output_file)
fi

#check if chromosome sizes has been provided and has correct ending
if [[ -z "$chrom_sizes_file" ]]; then
  echo "Chrom sizes file file not provided, exiting";
  exit 1;
fi
if [[ $chrom_sizes_file != *.chrom.sizes ]];then 
    echo "incorrect ending: chrom sizes file does not end on .chrom.sizes:";
    echo $chrom_sizes_file;
    exit 1;
fi

#converting gzipped fragments file to intermediary bed file
#sort and remove alternative chromosomes in the process
echo "Creating intermediary bed file..."
zcat $fragments_input_file | awk 'BEGIN{OFS="\t"} $1 ~ /^chr/ {print $1,$2,$3}' | sort -k 1,1 > ${output_dir}/${output_prefix}.bed && \


#calculating unnormalised coverage (intermediate bed graph file, rpm normalisation by default)
echo "Calculating unnormalised coverage..."
bedtools genomecov -i ${output_dir}/${output_prefix}.bed -g $chrom_sizes_file -bg > ${output_dir}/${output_prefix}.bg
#removing intermediary bed file
rm ${output_dir}/${output_prefix}.bed

#converting intermediate bedgraph to bigwig
echo "Generating bigwig..."
bedGraphToBigWig ${output_dir}/${output_prefix}.bg $chrom_sizes_file ${output_dir}/${output_prefix}.bw
#removing intermediary bed graph file
rm ${output_dir}/${output_prefix}.bg 

#script has passed sanity checks
