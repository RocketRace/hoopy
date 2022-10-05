from opera import tokens, transform


def test_transform_operator_objects_inplace():
    src = "(a)(...)(not in)($$)(())(+)(2+)(~~)(+ -)(is*)"
    exp = "(a)(...) __operator_nonce_6e6f7420696e  __operator_nonce_2424 (()) __operator_nonce_2b (2+) __operator_nonce_7e7e (+ -)(is*)"
    toks = list(tokens.lex(src))
    transform.transform_operator_objects_inplace(toks, "nonce")
    out = tokens.unlex(toks)
    assert exp == out


def test_transform_infix_identifiers_inplace():
    src = "abcd  `efg` `123` 1`b`c`2 `  `"
    exp = "abcd  @ `123` 1@c`2 `  `"
    toks = list(tokens.lex(src))
    spans = transform.transform_infix_identifiers_inplace(toks)
    exp_spans = {
        transform.OperatorSpan(1, 4, 1, 8): "efg",
        transform.OperatorSpan(1, 15, 1, 16): "b",
    }
    out = tokens.unlex(toks)
    assert exp == out
    assert spans == exp_spans
