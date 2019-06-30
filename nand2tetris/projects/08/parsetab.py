
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'ADD AND ARGUMENT BRANCH_LABEL CALL CONSTANT EQ FUNCTION GOTO GT IF_GOTO LABEL LOCAL LT NEG NOT NUMBER OR POINTER POP PUSH RETURN STATIC SUB TEMP THAT THISstmt : stmt stmtstmt : LABEL BRANCH_LABELstmt : GOTO BRANCH_LABELstmt : IF_GOTO BRANCH_LABELstmt : EQ\n            | GT\n            | LT\n    stmt : NEG\n            | NOT\n    stmt : ADD\n            | SUB\n            | OR\n            | AND\n    stmt : PUSH LOCAL NUMBER\n            | PUSH ARGUMENT NUMBER\n            | PUSH THIS NUMBER\n            | PUSH THAT NUMBER\n            | PUSH TEMP NUMBER\n            | PUSH CONSTANT NUMBER\n            | PUSH POINTER NUMBER\n            | PUSH STATIC NUMBER\n    stmt : POP LOCAL NUMBER\n            | POP ARGUMENT NUMBER\n            | POP THIS NUMBER\n            | POP THAT NUMBER\n            | POP TEMP NUMBER\n            | POP CONSTANT NUMBER\n            | POP POINTER NUMBER\n            | POP STATIC NUMBER\n    stmt : FUNCTION BRANCH_LABEL NUMBERstmt : RETURNstmt : CALL BRANCH_LABEL NUMBER'
    
_lr_action_items = {'LABEL':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[2,2,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,2,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'GOTO':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[3,3,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,3,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'IF_GOTO':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[4,4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,4,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'EQ':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[5,5,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,5,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'GT':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[6,6,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,6,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'LT':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[7,7,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,7,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'NEG':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[8,8,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,8,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'NOT':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[9,9,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,9,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'ADD':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[10,10,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,10,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'SUB':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[11,11,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,11,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'OR':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[12,12,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,12,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'AND':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[13,13,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,13,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'PUSH':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[14,14,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,14,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'POP':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[15,15,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,15,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'FUNCTION':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[16,16,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,16,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'RETURN':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[17,17,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,17,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'CALL':([0,1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[18,18,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,18,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'$end':([1,5,6,7,8,9,10,11,12,13,17,19,20,21,22,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,],[0,-5,-6,-7,-8,-9,-10,-11,-12,-13,-31,-1,-2,-3,-4,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-32,]),'BRANCH_LABEL':([2,3,4,16,18,],[20,21,22,39,40,]),'LOCAL':([14,15,],[23,31,]),'ARGUMENT':([14,15,],[24,32,]),'THIS':([14,15,],[25,33,]),'THAT':([14,15,],[26,34,]),'TEMP':([14,15,],[27,35,]),'CONSTANT':([14,15,],[28,36,]),'POINTER':([14,15,],[29,37,]),'STATIC':([14,15,],[30,38,]),'NUMBER':([23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,],[41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'stmt':([0,1,19,],[1,19,19,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> stmt","S'",1,None,None,None),
  ('stmt -> stmt stmt','stmt',2,'p_program','VMTranslator.py',411),
  ('stmt -> LABEL BRANCH_LABEL','stmt',2,'p_label','VMTranslator.py',415),
  ('stmt -> GOTO BRANCH_LABEL','stmt',2,'p_goto','VMTranslator.py',421),
  ('stmt -> IF_GOTO BRANCH_LABEL','stmt',2,'p_if_goto','VMTranslator.py',427),
  ('stmt -> EQ','stmt',1,'p_bool_op','VMTranslator.py',433),
  ('stmt -> GT','stmt',1,'p_bool_op','VMTranslator.py',434),
  ('stmt -> LT','stmt',1,'p_bool_op','VMTranslator.py',435),
  ('stmt -> NEG','stmt',1,'p_unary_op','VMTranslator.py',442),
  ('stmt -> NOT','stmt',1,'p_unary_op','VMTranslator.py',443),
  ('stmt -> ADD','stmt',1,'p_binary_op','VMTranslator.py',450),
  ('stmt -> SUB','stmt',1,'p_binary_op','VMTranslator.py',451),
  ('stmt -> OR','stmt',1,'p_binary_op','VMTranslator.py',452),
  ('stmt -> AND','stmt',1,'p_binary_op','VMTranslator.py',453),
  ('stmt -> PUSH LOCAL NUMBER','stmt',3,'p_push_op','VMTranslator.py',460),
  ('stmt -> PUSH ARGUMENT NUMBER','stmt',3,'p_push_op','VMTranslator.py',461),
  ('stmt -> PUSH THIS NUMBER','stmt',3,'p_push_op','VMTranslator.py',462),
  ('stmt -> PUSH THAT NUMBER','stmt',3,'p_push_op','VMTranslator.py',463),
  ('stmt -> PUSH TEMP NUMBER','stmt',3,'p_push_op','VMTranslator.py',464),
  ('stmt -> PUSH CONSTANT NUMBER','stmt',3,'p_push_op','VMTranslator.py',465),
  ('stmt -> PUSH POINTER NUMBER','stmt',3,'p_push_op','VMTranslator.py',466),
  ('stmt -> PUSH STATIC NUMBER','stmt',3,'p_push_op','VMTranslator.py',467),
  ('stmt -> POP LOCAL NUMBER','stmt',3,'p_pop_op','VMTranslator.py',474),
  ('stmt -> POP ARGUMENT NUMBER','stmt',3,'p_pop_op','VMTranslator.py',475),
  ('stmt -> POP THIS NUMBER','stmt',3,'p_pop_op','VMTranslator.py',476),
  ('stmt -> POP THAT NUMBER','stmt',3,'p_pop_op','VMTranslator.py',477),
  ('stmt -> POP TEMP NUMBER','stmt',3,'p_pop_op','VMTranslator.py',478),
  ('stmt -> POP CONSTANT NUMBER','stmt',3,'p_pop_op','VMTranslator.py',479),
  ('stmt -> POP POINTER NUMBER','stmt',3,'p_pop_op','VMTranslator.py',480),
  ('stmt -> POP STATIC NUMBER','stmt',3,'p_pop_op','VMTranslator.py',481),
  ('stmt -> FUNCTION BRANCH_LABEL NUMBER','stmt',3,'p_function','VMTranslator.py',488),
  ('stmt -> RETURN','stmt',1,'p_return','VMTranslator.py',494),
  ('stmt -> CALL BRANCH_LABEL NUMBER','stmt',3,'p_call','VMTranslator.py',500),
]
