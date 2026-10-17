"""Export, verify or install AudioLab weights without third-party packages."""
import argparse
import json
import zipfile

from src.product.model_bundle import export_bundle, install_bundle, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='package the three existing weights')
    export.add_argument('--model-root', default='checkpoints/timing_v1')
    export.add_argument('--output', required=True)
    export.add_argument('--bundle-id', required=True)
    for command in ('verify', 'install'):
        child = commands.add_parser(command)
        child.add_argument('--bundle', required=True)
        child.add_argument('--sha256', help='expected SHA-256 from the source PC or pinned manifest')
        if command == 'install':
            child.add_argument('--model-root', default='checkpoints/timing_v1')
    args = parser.parse_args()
    try:
        if args.command == 'export':
            result = export_bundle(args.model_root, args.output, bundle_id=args.bundle_id)
        elif args.command == 'verify':
            result = verify_bundle(args.bundle, expected_sha256=args.sha256)
        else:
            result = install_bundle(args.bundle, args.model_root, expected_sha256=args.sha256)
    except (ValueError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Model bundle: {exc}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
