#!/usr/bin/env python3
"""Regenerate a portable HTML report without re-running optical calculations."""
import argparse
from reporting import write_html_report, write_sms_report, write_haze_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', required=True, help='Existing results directory')
    parser.add_argument('--kind', choices=('brightness', 'sms-reference', 'haze'), default='brightness',
                        help='Explicit metric: raw brightness CV, separated sparkle, or saved native-backend haze comparison')
    parser.add_argument('--out', help='HTML destination; SMS mode copies its overview here')
    parser.add_argument('--recursive', action='store_true', help='Combine nested run directories')
    parser.add_argument('--title', help='Custom title for a brightness report')
    parser.add_argument('--figure', action='append', default=[], help='Embed a comparison PNG; repeat for multiple figures')
    args = parser.parse_args()
    if args.kind != 'brightness':
        if args.recursive or args.title or args.figure:
            parser.error('--recursive, --title and --figure apply only to --kind brightness')
        output = (write_sms_report(args.results, args.out) if args.kind == 'sms-reference' else
                  write_haze_report(args.results, args.out))
    else:
        output = write_html_report(args.results, args.out, args.recursive,
                                   args.title or 'OLED / AG 亮度计算报告', args.figure)
    print('HTML report:', output)


if __name__ == '__main__':
    main()
