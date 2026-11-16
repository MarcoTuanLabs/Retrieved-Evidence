"""Synthetic behavioral checks before opening any new comparator quality results."""
import json
from h29_comparator import parse_question, decide, date_interval


def doc(title, body, suffix=''):
    return dict(doc_id=title+suffix, title=title, body=body)


def main():
    checks = []
    def check(name, q, docs, expected):
        result = decide(parse_question(q), docs)
        assert result.get('answer') == expected, (name, result)
        checks.append(name)
    people = [doc('Ada Blue','Ada Blue (born 2 March 1980) is a musician.'),
              doc('Ben Green','Ben Green (3 March 1980 – 1 June 2020) was a musician.')]
    check('day_order', 'Who was born earlier, Ada Blue or Ben Green?', people, 'Ada Blue')
    check('younger_direction', 'Who is younger, Ada Blue or Ben Green?', people, 'Ben Green')
    check('missing_death_abstains', 'Who died later, Ada Blue or Ben Green?', people, None)
    check('unsupported_relation_abstains', 'Where was Ada Blue born?', people, None)
    check('missing_entity_abstains', 'Who is younger, Ada Blue or Ben Green?', people[:1], None)
    check('duplicate_title_abstains', 'Who is younger, Ada Blue or Ben Green?', people+[doc('Ada Blue',people[0]['body'],'2')], None)
    check('year_uncertainty_abstains', 'Who was born earlier, Ada Blue or Ben Green?', [doc('Ada Blue','Ada Blue (born 1980) is a musician.'),people[1]], None)
    films = [doc('Alpha','Alpha is a 1980 drama film directed by Ada Blue and starring Pat Brown.'),
             doc('Beta','Beta is a 1990 drama film directed by Ben Green.')]
    check('film_release', 'Which film was released more recently, Alpha or Beta?', films, 'Beta')
    check('director_chain', 'Which film has the director who was born earlier, Alpha or Beta?', films+people, 'Alpha')
    check('codirector_abstains', 'Which film has the director born later, Alpha or Beta?',
          [doc('Alpha','Alpha is a 1980 film directed by Ada Blue and Ben Green.'),films[1]]+people, None)
    check('false_body_identity_abstains', 'Who is younger, Ada Blue or Ben Green?',
          [doc('Ada Blue','Somebody Else (born 1970) is a singer.'),people[1]], None)
    check('no_borrowed_year', 'Which film came out first, Alpha or Beta?',
          [doc('Alpha','Alpha is a film based on a 1910 novel. It opened in 2000.'),films[1]], None)
    check('tie_abstains', 'Who was born first, Ada Blue or Ben Green?',
          [people[0],doc('Ben Green','Ben Green (born 2 March 1980) is a musician.')],None)
    check('was_question', 'Was Ada Blue or Ben Green born first?', people, 'Ada Blue')
    assert date_interval('February 30, 1980') is None
    assert date_interval('c. 1800') is None
    assert date_interval('1980 or 1981') is None
    assert date_interval('March 1980')['hi'] == '1980-03-31'
    print(json.dumps(dict(status='passed', behavioral_checks=checks, date_checks=4)))


if __name__ == '__main__': main()
