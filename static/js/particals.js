/**
 * Particleground demo
 * @author Jonathan Nicol - @mrjnicol
 */

$(document).ready(function() {
  $('#particles').particleground({
    dotColor: '#ed3a2c',
    lineColor: '#fab12f'
  });
  $('.intro').css({
    'margin-top': -($('.intro').height() / 2)
  });
});