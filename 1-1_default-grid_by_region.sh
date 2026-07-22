#!/bin/bash
# chmod +x 1-1_default_grid.sh
# ./1-1_default-grid_by_region.sh --world_size 1 --pts_per_batch 32 --overlay_only True --start_img_idx 1600 --end_img_idx 2000 --domain_name mask2former
# ./1-1_default-grid_by_region.sh --world_size 1 --pts_per_batch 32 --overlay_only False --start_img_idx 0 --end_img_idx 1000 --domain_name mask2former
# ./1-1_default-grid_by_region.sh --world_size 1 --pts_per_batch 32 --overlay_only False --start_img_idx 1000 --end_img_idx 2000 --domain_name ground-truth

# Initialize default world_size
world_size=1
pts_per_batch=32
overlay_only=True
start_img_idx=0
end_img_idx=2000
domain_name="mask2former"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --world_size)
            world_size="$2"
            shift 2
            ;;
        --pts_per_batch)
            pts_per_batch="$2"
            shift 2
            ;;
        --overlay_only)
            overlay_only="$2"
            shift 2
            ;;
        --start_img_idx)
            start_img_idx="$2"
            shift 2
            ;;
        --end_img_idx)
            end_img_idx="$2"
            shift 2
            ;;
        --domain_name)
            domain_name="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate numerical arguments
validate_positive_int() {
    local name="$1"
    local value="$2"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: $name must be a positive integer, got '$value'" >&2
        exit 1
    fi
}

validate_positive_int "world_size" "$world_size"
validate_positive_int "pts_per_batch" "$pts_per_batch"

params=(
    "sam_siz vit_h"
    "pts_per_side 128"
    "crop_config 0:1.0"
    "uncertainty_strategy relative_0.2-0.5"
)

# Generate parameter combinations using nested loops
generate_combinations() {
    local combinations=("")
    
    for param in "${params[@]}"; do
        IFS=' ' read -r key values <<< "$param"
        local new_combinations=()
        
        for combination in "${combinations[@]}"; do
            for value in $values; do
                # Special handling for crop_config
                if [[ $key == "crop_config" ]]; then
                    IFS=':' read -r crop_layer crop_scale <<< "$value"
                    new_combinations+=("${combination}crop_n_layers:${crop_layer} crop_n_points_downscale_factor:${crop_scale} ")
                else
                    new_combinations+=("${combination}${key}:${value} ")
                fi
            done
        done
        
        combinations=("${new_combinations[@]}")
    done
    
    printf '%s\n' "${combinations[@]}"
}

# Main execution loop
generate_combinations | while read -r combination; do
    # Build command arguments
    args=()
    for pair in $combination; do
        IFS=':' read -r key value <<< "$pair"
        args+=(--$key $value)
    done
    
    echo "Running: python3 1-1_default-grid_by_region.py ${args[*]}"
    # Actual execution would use:
    python3 1-1_default-grid_by_region.py "${args[@]}" \
        --world_size $world_size \
        --data_dir "../data/mapillary_v1" \
        --data_grp validation \
        --start_img_idx $start_img_idx \
        --end_img_idx $end_img_idx \
        --sam_author hugging_face \
        --pts_per_batch $pts_per_batch \
        --domain_name $domain_name \
        --overlay_only $overlay_only
done

echo "All combinations executed."