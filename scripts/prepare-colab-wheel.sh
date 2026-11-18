#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ || $1 -lt 1 || $1 -gt 604800 ]]; then
    echo "Usage: $0 URL_EXPIRY_SECONDS (1-604800)" >&2
    exit 2
fi
expires=$1
for command in aws uv realpath shasum tar find sort; do
    command -v "$command" >/dev/null || { echo "Missing command: $command" >&2; exit 1; }
done

project_dir=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$project_dir"
[[ -n ${R2_BUCKET:-} && -n ${R2_CREDENTIALS_FILE:-} ]] || {
    echo "Set R2_BUCKET and R2_CREDENTIALS_FILE before preparing the Worldsmith wheel." >&2
    exit 1
}
credentials_file=$(realpath "$R2_CREDENTIALS_FILE")
case "$credentials_file" in
    "$project_dir"|"$project_dir"/*)
        echo "R2_CREDENTIALS_FILE must be outside the Worldsmith repository." >&2
        exit 1
        ;;
esac

account_id=
access_key=
secret_key=
while IFS='=' read -r key value || [[ -n $key ]]; do
    case "$key" in
        R2_ACCOUNT_ID) account_id=$value ;;
        R2_ACCESS_KEY_ID) access_key=$value ;;
        R2_SECRET_ACCESS_KEY) secret_key=$value ;;
    esac
done < "$credentials_file"
[[ $account_id =~ ^[0-9a-fA-F]{32}$ && -n $access_key && -n $secret_key ]] || {
    echo "Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY in the credential file." >&2
    exit 1
}
export AWS_ACCESS_KEY_ID="$access_key" AWS_SECRET_ACCESS_KEY="$secret_key"
unset AWS_SESSION_TOKEN AWS_SECURITY_TOKEN
export AWS_DEFAULT_REGION=auto AWS_EC2_METADATA_DISABLED=true AWS_MAX_ATTEMPTS=1 AWS_PAGER=
export AWS_REQUEST_CHECKSUM_CALCULATION=when_required
endpoint="https://${account_id}.r2.cloudflarestorage.com"

temporary_dir=$(mktemp -d)
trap 'rm -rf "$temporary_dir"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Hash the selected source bytes and paths. Candidates and installed environments
# are outside this file list, and the script itself versions the build procedure.
{
    find worldsmith -type f -name '*.py' -print0
    printf '%s\0' pyproject.toml worldsmith/research_baseline/JOURNAL.md scripts/prepare-colab-wheel.sh
} | LC_ALL=C sort -z > "$temporary_dir/files"
digest=$(
    {
        printf 'worldsmith-wheel-v2\0'
        while IFS= read -r -d '' file; do
            printf '%s\0' "$file"
            shasum -a 256 "$file" | cut -d ' ' -f 1
        done < "$temporary_dir/files"
    } | shasum -a 256 | cut -d ' ' -f 1
)
prefix="worldsmith/cache/package/$digest"
if ! cached_keys=$(aws --endpoint-url "$endpoint" s3api list-objects-v2 \
    --bucket "$R2_BUCKET" --prefix "$prefix/" --query 'Contents[].Key' --output text \
    2>"$temporary_dir/error"); then
    echo "R2 wheel cache check failed." >&2
    exit 1
fi

if [[ -z $cached_keys || $cached_keys == None ]]; then
    mkdir "$temporary_dir/source" "$temporary_dir/wheels"
    COPYFILE_DISABLE=1 tar --null -T "$temporary_dir/files" -cf "$temporary_dir/source.tar"
    tar -xf "$temporary_dir/source.tar" -C "$temporary_dir/source"
    if ! SOURCE_DATE_EPOCH=315532800 uv build --wheel --out-dir "$temporary_dir/wheels" \
        "$temporary_dir/source" >"$temporary_dir/build.log" 2>&1; then
        echo "Worldsmith wheel build failed." >&2
        exit 1
    fi
    wheels=("$temporary_dir/wheels/"*.whl)
    [[ ${#wheels[@]} -eq 1 && -f ${wheels[0]} ]] || {
        echo "Worldsmith build must produce one wheel." >&2
        exit 1
    }
    wheel=${wheels[0]}
    filename=${wheel##*/}
    cache_key="$prefix/$filename"
    if ! aws --endpoint-url "$endpoint" s3api put-object --bucket "$R2_BUCKET" \
        --key "$cache_key" --body "$wheel" --if-none-match '*' \
        >"$temporary_dir/upload.json" 2>"$temporary_dir/error"; then
        # Another startup can publish the same source concurrently.
        if ! grep -q 'PreconditionFailed' "$temporary_dir/error"; then
            echo "R2 wheel upload failed." >&2
            exit 1
        fi
    fi
else
    cache_key=$cached_keys
    filename=${cache_key##*/}
    [[ $cache_key == "$prefix/"* && $filename == worldsmith-*-py3-none-any.whl && $cached_keys != *$'\t'* && $cached_keys != *$'\n'* ]] || {
        echo "R2 wheel cache contains unexpected objects." >&2
        exit 1
    }
fi

if ! url=$(aws --endpoint-url "$endpoint" s3 presign "s3://$R2_BUCKET/$cache_key" \
    --expires-in "$expires" 2>"$temporary_dir/error"); then
    echo "R2 wheel URL signing failed." >&2
    exit 1
fi
# The caller captures these two lines. Keep signed URLs out of diagnostic logs.
printf '%s\n%s\n' "$filename" "$url"
