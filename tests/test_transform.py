from opera import tokens, transform


def test_transform_operator_objects_inplace():
    src = "(a)(...)(not in)($$)(())(+)(2+)(~~)(+ -)(is*)"
    exp = "(a)(...) __operator_nonce_6e6f7420696e  __operator_nonce_2424 (()) __operator_nonce_2b (2+) __operator_nonce_7e7e (+ -)(is*)"
    toks = list(tokens.lex(src))
    transform.transform_operator_objects_inplace(toks, "nonce")
    out = tokens.unlex(toks)
    assert exp == out
